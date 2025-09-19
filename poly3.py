#!/usr/bin/env python3
"""
Advanced Go obfuscator with evasion techniques + sensitive string encryption
- Enhanced identifier obfuscation with multiple strategies
- Advanced code transformation layers
- Anti-debugging and anti-analysis techniques
- Polymorphic code generation
- Steganography and encryption layers
- Config-driven encryption of sensitive strings (URLs, IPs, API endpoints...)
"""

import re
import random
import string
import hashlib
import time
import os
import sys
import threading
import subprocess
import base64
import json
from cryptography.fernet import Fernet

# --- Config / reserved sets ---
GO_KEYWORDS = {
    "break","case","chan","const","continue","default","defer","else","fallthrough",
    "for","func","go","goto","if","import","interface","map","package","range",
    "return","select","struct","switch","type","var",
}

PREDECLARED = {
    "append","cap","close","complex","copy","delete","imag","len","make","new",
    "panic","print","println","real","recover",
    "bool","byte","complex64","complex128","error","float32","float64","int",
    "int8","int16","int32","int64","rune","string","uint","uint8","uint16","uint32","uint64","uintptr",
    "true","false","iota","nil",
}

IDENT_START = re.compile(r'[A-Za-z_]')
IDENT_BODY = re.compile(r'[A-Za-z0-9_]*')

# Advanced obfuscation strategies
OBFUSCATION_STRATEGIES = [
    'random', 'leet', 'unicode', 'hex', 'base64', 'reverse'
]

def generate_encryption_key():
    return Fernet.generate_key()

def encrypt_code(code, key):
    fernet = Fernet(key)
    encrypted = fernet.encrypt(code.encode())
    return base64.b64encode(encrypted).decode()

def decrypt_code(encrypted_code, key):
    fernet = Fernet(key)
    decoded = base64.b64decode(encrypted_code.encode())
    return fernet.decrypt(decoded).decode()

def random_name(strategy='random', length=8):
    strategies = {
        'random': lambda: ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length)),
        'leet': lambda: ''.join(random.choice('4BCD3FGH1JKLMN0PQRST7VWXYZ2bcd3fgh1jklmn0pqrst7vwxyz') for _ in range(length)),
        'unicode': lambda: ''.join(chr(random.randint(0x200, 0x2FF)) for _ in range(length//2)) +
                           ''.join(random.choice(string.ascii_lowercase) for _ in range(length//2)),
        'hex': lambda: 'x' + ''.join(random.choice('0123456789abcdef') for _ in range(length-1)),
        'base64': lambda: base64.b64encode(os.urandom(length)).decode()[:length].lower(),
        'reverse': lambda: ''.join(random.choice(string.ascii_lowercase) for _ in range(length))[::-1]
    }
    if strategy == 'random':
        strategy = random.choice(list(strategies.keys()))
    name = strategies[strategy]()
    if not name[0].isalpha() and name[0] != '_':
        name = random.choice(string.ascii_lowercase) + name[1:]
    return name

def calculate_signature(code):
    return hashlib.sha256(code.encode()).hexdigest()[:32]

# ---------------------------
# NEW: Sensitive string encryption
# ---------------------------

def encrypt_sensitive_strings(code, config_file, key):
    """Encrypt sensitive strings from config and replace them with decrypt() calls"""
    if not config_file or not os.path.exists(config_file):
        return code

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[!] Failed to load config: {e}")
        return code

    sensitive_strings = config.get("sensitive_strings", [])
    if not sensitive_strings:
        return code

    fernet = Fernet(key)
    replacements = {}

    for s in sensitive_strings:
        encrypted = fernet.encrypt(s.encode()).decode()
        replacements[s] = encrypted

    for original, encrypted in replacements.items():
        code = code.replace(f'"{original}"', f'decrypt("{encrypted}")')

    # Inject Go decrypt helper if missing
    if "func decrypt(" not in code:
        decrypt_func = '''
import (
    "encoding/base64"
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "fmt"
    "os"
)

func decrypt(enc string) string {
    key := []byte(os.Getenv("APP_KEY"))
    data, _ := base64.StdEncoding.DecodeString(enc)
    block, err := aes.NewCipher(key)
    if err != nil { return "" }
    gcm, err := cipher.NewGCM(block)
    if err != nil { return "" }
    nonceSize := gcm.NonceSize()
    if len(data) < nonceSize { return "" }
    nonce, ciphertext := data[:nonceSize], data[nonceSize:]
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil { return "" }
    return string(plaintext)
}
'''
        insertion_point = code.find("package main")
        if insertion_point != -1:
            code = code[:insertion_point] + "package main\n\n" + decrypt_func + "\n" + code[insertion_point+len("package main"):]
    return code

# ---------------------------
# Advanced obfuscation layers
# ---------------------------

def insert_anti_debug_code(code):
    anti_debug_snippets = [
        """// Anti-debug: Check if running under debugger
func isDebuggerPresent() bool {
    return false
}""",
        """// Anti-tamper: Integrity check
func integrityCheck() {
    // Hash of critical code sections could be verified here
}""",
        """// Anti-VM: Basic VM detection
func isRunningInVM() bool {
    return false
}"""
    ]
    insertion_point = code.find("func main")
    if insertion_point != -1:
        anti_code = "\n".join(random.sample(anti_debug_snippets, random.randint(1, len(anti_debug_snippets))))
        code = code[:insertion_point] + anti_code + "\n\n" + code[insertion_point:]
    return code

def polymorphic_code_transforms(code):
    transforms = [
        (r'if\s+([^{]+)\s*{', lambda m: f'if {m.group(1)} {{ /* polymorphic */ }}'),
        (r'for\s+([^{]+)\s*{', lambda m: f'for {m.group(1)} {{ /* obfuscated loop */ }}'),
        (r'"([^"]+)"', lambda m: f'string([]byte{{{", ".join([str(ord(c)) for c in m.group(1)])}}})'
                               if random.random() < 0.3 else m.group(0)),
    ]
    for pattern, replacement in transforms:
        code = re.sub(pattern, replacement, code)
    return code

def insert_decoy_functions(code, count=5):
    decoy_templates = [
        "func {name}() {{\n    {vars}\n    {operations}\n    {return_stmt}\n}}",
        "func {name}(args ...interface{}) {{\n    {vars}\n    {operations}\n    {return_stmt}\n}}"
    ]
    operations = [
        "x := make([]byte, 1024)",
        "rand.Read(x)",
        "time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)"
    ]
    variables = ["var x int","var data []byte","var result string"]
    return_statements = ["return","return nil","return 0"]
    decoys = []
    for _ in range(count):
        template = random.choice(decoy_templates)
        decoy = template.format(
            name=random_name('leet', random.randint(6, 12)),
            vars='\n    '.join(random.sample(variables, random.randint(1, 3))),
            operations='\n    '.join(random.sample(operations, random.randint(1, 3))),
            return_stmt=random.choice(return_statements)
        )
        decoys.append(decoy)
    main_index = code.find("func main")
    if main_index != -1:
        code = code[:main_index] + "\n".join(decoys) + "\n\n" + code[main_index:]
    return code

def obfuscate_string_literals(code):
    # Do NOT touch import paths
    safe_code = []
    inside_import = False

    for line in code.splitlines():
        if line.strip().startswith("import"):
            inside_import = True
        if inside_import and line.strip().endswith(")"):
            inside_import = False

        if inside_import:
            safe_code.append(line)  # leave imports untouched
        else:
            # existing string obfuscation logic here
            safe_code.append(transform_strings_in_line(line))

    return "\n".join(safe_code)

def add_metadata_steganography(code):
    stego_data = [
        f"// Checksum: {hashlib.md5(code.encode()).hexdigest()}",
        f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "// Build: go build -ldflags='-s -w'"
    ]
    lines = code.split('\n')
    insert_pos = random.randint(5, min(20, len(lines)))
    lines[insert_pos:insert_pos] = random.sample(stego_data, random.randint(1, len(stego_data)))
    return '\n'.join(lines)

def randomize_whitespace_advanced(code):
    code = re.sub(r' {2,}', lambda m: random.choice(['\t','  ','   ']), code)
    lines = code.split('\n')
    for i in range(len(lines)):
        if random.random() < 0.05 and lines[i].strip():
            lines[i] = lines[i] + '\n' + ''.join(random.choice([' ', '\t']) for _ in range(random.randint(1, 8)))
    return '\n'.join(lines)

def insert_random_comments(code, prob=0.18):
    lines = code.splitlines()
    out = []
    for line in lines:
        if line.strip() and random.random() < prob:
            comment = "// " + ''.join(random.choice(string.ascii_letters) for _ in range(10))
            out.append(comment)
        out.append(line)
    return "\n".join(out)

def insert_dummy_functions(code, min_funcs=1, max_funcs=3):
    dummy_funcs = []
    for _ in range(random.randint(min_funcs, max_funcs)):
        fname = random_name()
        a, b = random.randint(1, 999), random.randint(1, 999)
        dummy_funcs.append(f"func {fname}() int {{ x := {a}; y := {b}; return x ^ y }}")
    insertion = "\n".join(dummy_funcs) + "\n\n"
    return re.sub(r'(?m)^func\s+main\b', insertion + "func main", code, count=1)

def parse_imports(code):
    imports = set()
    patterns = [r'import\s+\(([^)]+)\)', r'import\s+"([^"]+)"', r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+"([^"]+)"']
    for pattern in patterns:
        for match in re.finditer(pattern, code, re.DOTALL):
            if match.groups():
                import_text = match.group(1)
                for line in import_text.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        if '"' in line:
                            path = re.search(r'"([^"]+)"', line)
                            if path:
                                base = path.group(1).split('/')[-1]
                                if base:
                                    imports.add(base)
    return imports

def advanced_obfuscate_code(code, encryption_key=None, config_file=None):
    random.seed(time.time_ns() + os.getpid())
    import_names = parse_imports(code)
    obf_code = code
    if config_file:
        obf_code = encrypt_sensitive_strings(obf_code, config_file, encryption_key)
    layers = [
        insert_anti_debug_code,
        polymorphic_code_transforms,
        insert_decoy_functions,
        obfuscate_string_literals,
        add_metadata_steganography,
        randomize_whitespace_advanced,
        lambda x: insert_random_comments(x, 0.25),
        lambda x: insert_dummy_functions(x, 2, 5)
    ]
    random.shuffle(layers)
    for layer in layers:
        try:
            obf_code = layer(obf_code)
        except Exception as e:
            print(f"[!] Layer {layer.__name__} failed: {e}")
    info = {'signature': calculate_signature(obf_code), 'imports': list(import_names), 'functions_found': 0}
    return obf_code, info

def transform_go_advanced(input_file, output_file, morph_interval=30, config_file=None):
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)
    with open(input_file, 'r', encoding='utf-8') as f:
        original_code = f.read()
    encryption_key = generate_encryption_key()
    def morph_task():
        nonlocal encryption_key
        while True:
            time.sleep(morph_interval)
            try:
                if random.random() < 0.1:
                    encryption_key = generate_encryption_key()
                obf_code, info = advanced_obfuscate_code(original_code, encryption_key, config_file)
                header = f"""// Auto-obfuscated by advanced_go_obfuscator.py
// Signature: {info['signature']}
// Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
// Protection: multi-layer evasion techniques
//
"""
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(header + obf_code)
                print(f"[~] Advanced morph completed at {time.strftime('%H:%M:%S')}")
                print(f"    Signature: {info['signature']}")
            except Exception as e:
                print(f"[!] Advanced morphing error: {e}")
    t = threading.Thread(target=morph_task, daemon=True)
    t.start()
    try:
        print("[+] Compiling obfuscated code...")
        compile_proc = subprocess.run(["go", "build", "-o", "obfuscated_binary", output_file],
                                      capture_output=True, text=True)
        if compile_proc.returncode == 0:
            print("[+] Running obfuscated binary...")
            run_proc = subprocess.Popen(["./obfuscated_binary"])
            print(f"[+] Program running with PID {run_proc.pid}")
        else:
            print("[!] Compilation failed:")
            print(compile_proc.stderr)
    except Exception as e:
        print(f"[!] Failed to launch: {e}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")

def main():
    if len(sys.argv) < 3:
        print("Usage: python advanced_go_obfuscator.py input.go output.go [morph_interval] [config.json]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    morph_interval = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 30
    config_file = sys.argv[4] if len(sys.argv) > 4 else None
    transform_go_advanced(input_file, output_file, morph_interval, config_file)

if __name__ == '__main__':
    main()
