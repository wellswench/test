#!/usr/bin/env python3
"""
Go obfuscator (Python)
- Renames unexported function names + function-local identifiers (params, locals, loop vars, named returns).
- Does NOT rename imports, package, or exported (Uppercase) symbols.
- Periodically re-randomizes identifiers in the output file.
- Automatically launches the obfuscated Go program once and keeps it running while the file keeps morphing.
- NEW: Adds additional obfuscation layers (random comments, dummy functions, whitespace randomization)
Usage:
    python go_obfuscator.py input.go output.go [morph_interval_seconds]
"""

import re
import random
import string
import hashlib
import time
import os
import sys
import threading
import json
import base64
import subprocess

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

CONFIG_FILE = "conf.json"

def load_config():
    """Load sensitive strings from config.json if available."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("sensitive_strings", []))
        except Exception as e:
            print(f"[!] Failed to load {CONFIG_FILE}: {e}")
            return set()
    return set()

SENSITIVE_STRINGS = load_config()

def random_name(length=8):
    first = random.choice(string.ascii_lowercase)
    rest = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length - 1))
    return first + rest

def calculate_signature(code):
    return hashlib.md5(code.encode()).hexdigest()[:16]


# ---------------------------
# String encryption support
# ---------------------------

def xor_encrypt(s, key=0x5A):
    b = bytes([c ^ key for c in s.encode('utf-8')])
    return base64.b64encode(b).decode('utf-8')

def generate_decrypt_function():
    return """
func decryptString(b64 string) string {
    data, _ := base64.StdEncoding.DecodeString(b64)
    for i := 0; i < len(data); i++ {
        data[i] ^= 0x5A
    }
    return string(data)
}
"""

def encrypt_strings_in_code(code):
    """Find and replace sensitive strings with decryptString(base64data)."""
    out, i, ln = [], 0, len(code)
    used_encryption = False
    while i < ln:
        if code[i] == '"':
            j = i + 1
            sb = []
            while j < ln:
                if code[j] == '\\':
                    sb.append(code[j]); sb.append(code[j+1]); j += 2; continue
                if code[j] == '"':
                    break
                sb.append(code[j])
                j += 1
            literal = ''.join(sb)
            if literal in SENSITIVE_STRINGS:
                enc = xor_encrypt(literal)
                out.append(f'decryptString("{enc}")')
                used_encryption = True
                j += 1
                i = j
                continue
            out.append(code[i:j+1])
            i = j+1
            continue
        out.append(code[i])
        i += 1
    new_code = ''.join(out)
    if used_encryption:
        # 1) Ensure encoding/base64 is imported
        if 'encoding/base64' not in new_code:
            # Case 1: import block exists
            if re.search(r'(?m)^import\s*\(', new_code):
                new_code = re.sub(
                r'(?ms)^import\s*\((.*?)\)',
                lambda m: f'import ({m.group(1)}\n\t"encoding/base64")',
                new_code,
                count=1
            )
            # Case 2: single-line import (rare)
            elif re.search(r'(?m)^import\s+"[^"]+"', new_code):
                new_code = re.sub(
                    r'(?m)^import\s+"([^"]+)"',
                    r'import (\n\t"\1"\n\t"encoding/base64"\n)',
                    new_code,
                    count=1
                )
            # Case 3: no import block at all — create a new one
            else:
                new_code = re.sub(
                    r'(?m)^(package\s+\w+)',
                    r'\1\n\nimport "encoding/base64"',
                    new_code,
                    count=1
                )

        # 2) Insert decryptString AFTER the import block OR after package if no import block exists
        if re.search(r'(?ms)^import\s*\(', new_code):
            new_code = re.sub(
                r'(?ms)(^import\s*\(.*?\))',
                r'\1\n\n' + generate_decrypt_function().strip(),
                new_code,
                count=1
            )
        elif re.search(r'(?m)^import\s+"[^"]+"', new_code):
            # single-line import, insert after it
            new_code = re.sub(
                r'(?m)^(import\s+"[^"]+")',
                r'\1\n\n' + generate_decrypt_function().strip(),
                new_code,
                count=1
            )
        else:
            # no import block — insert after package declaration
            new_code = re.sub(
                r'(?m)^(package\s+\w+)',
                r'\1\n\n' + generate_decrypt_function().strip(),
                new_code,
                count=1
            )

        # 3) Convert const -> var for decryptString calls
        new_code = re.sub(
            r'(?m)^const(\s*\([^)]*decryptString\([^)]*\)[^)]*\))',
            r'var\1',
            new_code
        )
        new_code = re.sub(
            r'(?m)^const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(decryptString\([^)]*\))',
            r'var \1 = \2',
            new_code
        )


    return new_code


# ---------------------------
# NEW EXTRA LAYERS
# ---------------------------

def insert_random_comments(code, prob=0.18):
    """Insert random harmless comments in approximately `prob` fraction of non-blank lines."""
    lines = code.splitlines()
    out = []
    for line in lines:
        if line.strip() and random.random() < prob:
            comment = "// " + ''.join(random.choice(string.ascii_letters) for _ in range(10))
            out.append(comment)
        out.append(line)
    return "\n".join(out)

def insert_dummy_functions(code, min_funcs=1, max_funcs=3):
    """Insert a few dummy unused functions before the first 'func main' occurrence."""
    dummy_funcs = []
    for _ in range(random.randint(min_funcs, max_funcs)):
        fname = random_name()
        # create a slightly more interesting dummy function
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        dummy = f"func {fname}() int {{ x := {a}; y := {b}; return x ^ y }}"
        dummy_funcs.append(dummy)
    insertion = "\n".join(dummy_funcs) + "\n\n"
    return re.sub(r'(?m)^func\s+main\b', insertion + "func main", code, count=1)

def randomize_whitespace(code):
    """Randomize some whitespace (tabs/spaces/newlines) to change the file signature slightly."""
    # compress runs of spaces to a random smaller/bigger run
    code = re.sub(r' {2,}', lambda m: " " * random.randint(1, 4), code)
    # sometimes add a leading tab to lines
    code = re.sub(r'(?m)^(?P<ln>.*)$', lambda m: ("\t" + m.group('ln') if m.group('ln').strip() and random.random() < 0.07 else m.group('ln')), code)
    return code

# ---------------------------
# Parsing helpers
# ---------------------------

def parse_imports(code):
    imports = set()
    for m in re.finditer(r'(?m)^\s*import\s*\((.*?)\)', code, re.DOTALL):
        block = m.group(1)
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            ma = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s+"([^"]+)"', line)
            if ma:
                imports.add(ma.group(1))
                continue
            mp = re.match(r'"([^"]+)"', line)
            if mp:
                path = mp.group(1)
                base = path.split('/')[-1]
                if base:
                    imports.add(base)
                continue
    for m in re.finditer(r'(?m)^\s*import\s+([A-Za-z_][A-Za-z0-9_]*)\s+"([^"]+)"', code):
        imports.add(m.group(1))
    for m in re.finditer(r'(?m)^\s*import\s+"([^"]+)"', code):
        path = m.group(1)
        base = path.split('/')[-1]
        if base:
            imports.add(base)
    imports = {n for n in imports if n and n not in GO_KEYWORDS}
    return imports

def _skip_string_or_comment_forward(code, i):
    ln = len(code)
    if i >= ln:
        return i
    ch = code[i]
    if ch == '"':
        j = i + 1
        while j < ln:
            if code[j] == '\\':
                j += 2
                continue
            if code[j] == '"':
                return j + 1
            j += 1
        return ln
    if ch == '`':
        j = code.find('`', i+1)
        return j+1 if j != -1 else ln
    if ch == "'":
        j = i + 1
        while j < ln:
            if code[j] == '\\':
                j += 2
                continue
            if code[j] == "'":
                return j + 1
            j += 1
        return ln
    if code.startswith('//', i):
        j = code.find('\n', i+2)
        return j+1 if j != -1 else ln
    if code.startswith('/*', i):
        j = code.find('*/', i+2)
        return j+2 if j != -1 else ln
    return i

def find_matching_paren(code, start_idx):
    i = start_idx + 1
    depth = 1
    ln = len(code)
    while i < ln:
        if code[i] in ('"', '`', "'") or code.startswith('//', i) or code.startswith('/*', i):
            i = _skip_string_or_comment_forward(code, i)
            continue
        ch = code[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def find_matching_brace(code, start_idx):
    i = start_idx + 1
    depth = 1
    ln = len(code)
    while i < ln:
        if code[i] in ('"', '`', "'") or code.startswith('//', i) or code.startswith('/*', i):
            i = _skip_string_or_comment_forward(code, i)
            continue
        ch = code[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def skip_ws_and_comments(code, i):
    ln = len(code)
    while i < ln:
        if code[i].isspace():
            i += 1
            continue
        if code.startswith('//', i) or code.startswith('/*', i):
            i = _skip_string_or_comment_forward(code, i)
            continue
        break
    return i

def find_functions(code):
    funcs = []
    ln = len(code)
    i = 0
    while i < ln:
        if code[i] in ('"', '`', "'") or code.startswith('//', i) or code.startswith('/*', i):
            i = _skip_string_or_comment_forward(code, i)
            continue
        if code.startswith('func', i):
            before = code[i-1] if i-1 >= 0 else ' '
            after = code[i+4] if i+4 < ln else ' '
            if (not (before.isalnum() or before == '_')) and (not (after.isalnum() or after == '_')):
                func_start = i
                j = i + 4
                j = skip_ws_and_comments(code, j)
                receiver_start = None
                params_start = None
                params_end = None
                func_name = None
                if j < ln and code[j] == '(':
                    p1 = find_matching_paren(code, j)
                    if p1 == -1:
                        i += 4
                        continue
                    k = skip_ws_and_comments(code, p1+1)
                    if k < ln and IDENT_START.match(code[k]):
                        receiver_start = j
                        m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', code[k:])
                        if m:
                            func_name = m.group(0)
                            k = k + m.end()
                            k = skip_ws_and_comments(code, k)
                            if k < ln and code[k] == '(':
                                params_start = k
                                params_end = find_matching_paren(code, params_start)
                                if params_end == -1:
                                    i += 4
                                    continue
                                sig_end = params_end + 1
                            else:
                                sig_end = k
                        else:
                            sig_end = p1 + 1
                    else:
                        params_start = j
                        params_end = p1
                        sig_end = params_end + 1
                elif j < ln and IDENT_START.match(code[j]):
                    m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', code[j:])
                    func_name = m.group(0)
                    k = j + m.end()
                    k = skip_ws_and_comments(code, k)
                    if k < ln and code[k] == '(':
                        params_start = k
                        params_end = find_matching_paren(code, params_start)
                        if params_end == -1:
                            i += 4
                            continue
                        sig_end = params_end + 1
                    else:
                        sig_end = k
                else:
                    i += 4
                    continue
                body_search_idx = sig_end if 'sig_end' in locals() else j
                body_search_idx = skip_ws_and_comments(code, body_search_idx)
                if body_search_idx < ln and code[body_search_idx] == '{':
                    body_start = body_search_idx
                    body_end = find_matching_brace(code, body_start)
                    if body_end == -1:
                        i += 4
                        continue
                    funcs.append({
                        'func_start': func_start,
                        'sig_start': func_start,
                        'sig_end': body_start,
                        'body_start': body_start,
                        'body_end': body_end + 1,
                        'params_range': (params_start, params_end) if params_start is not None else None,
                        'func_name': func_name
                    })
                    i = body_end + 1
                    continue
                else:
                    i += 4
                    continue
        i += 1
    return funcs

def extract_param_and_named_returns(code, params_range, sig_range_end):
    params = set()
    named_returns = set()
    if not params_range:
        return params, named_returns
    ps, pe = params_range
    param_str = code[ps+1:pe]
    for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s+[^\s,)\[]+', param_str):
        names_group = m.group(1)
        for name in [n.strip() for n in names_group.split(',')]:
            if name and name != '_':
                params.add(name)
    idx_after_params = pe + 1
    idx_after_params = skip_ws_and_comments(code, idx_after_params)
    if idx_after_params < len(code) and code[idx_after_params] == '(':
        rstart = idx_after_params
        rend = find_matching_paren(code, rstart)
        if rend != -1:
            ret_str = code[rstart+1:rend]
            for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s+[^\s,)\[]+', ret_str):
                names_group = m.group(1)
                for name in [n.strip() for n in names_group.split(',')]:
                    if name and name != '_':
                        named_returns.add(name)
    return params, named_returns

def collect_locals_in_body(body_text):
    locals_set = set()
    for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*:=', body_text):
        names = [n.strip() for n in m.group(1).split(',')]
        for name in names:
            if name and name != '_':
                locals_set.add(name)
    for m in re.finditer(r'\bvar\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)', body_text):
        names = [n.strip() for n in m.group(1).split(',')]
        for name in names:
            if name and name != '_':
                locals_set.add(name)
    for m in re.finditer(r'\bfor\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*(?:[:=]{1,2})?\s*range\b', body_text):
        names = [n.strip() for n in m.group(1).split(',')]
        for name in names:
            if name and name != '_':
                locals_set.add(name)
    for m in re.finditer(r'\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s*:=', body_text):
        name = m.group(1)
        if name and name != '_':
            locals_set.add(name)
    return locals_set

def build_mappings_for_functions(code, funcs, import_names):
    mappings = []
    func_renames = {}
    reserved = GO_KEYWORDS.union(PREDECLARED).union(import_names)
    for f in funcs:
        params_range = f.get('params_range')
        params, nreturns = extract_param_and_named_returns(code, params_range, f['sig_end'])
        body_text = code[f['body_start']:f['body_end']]
        locals_in_body = collect_locals_in_body(body_text)
        mapping = {}
        names = params | nreturns | locals_in_body
        for name in names:
            if name and name not in reserved and name != '_' and not name[0].isupper():
                new_name = random_name()
                while new_name in reserved or new_name in mapping.values():
                    new_name = random_name()
                mapping[name] = new_name
        fname = f.get('func_name')
        if fname and fname not in reserved and fname != "main" and not fname[0].isupper():
            new_fname = random_name()
            while new_fname in func_renames.values():
                new_fname = random_name()
            func_renames[fname] = new_fname
        mappings.append(mapping)
    return mappings, func_renames

def find_innermost_function(funcs, pos):
    inn = None
    for idx, f in enumerate(funcs):
        if f['sig_start'] <= pos < f['body_end']:
            if inn is None or f['sig_start'] > funcs[inn]['sig_start']:
                inn = idx
    return inn

def obfuscate_code(code):
    import time
    random.seed(time.time_ns())  # NEW: re-seed RNG for every morph pass
    import_names = parse_imports(code)
    code = encrypt_strings_in_code(code)
    funcs = find_functions(code)
    mappings, func_renames = build_mappings_for_functions(code, funcs, import_names)
    reserved = GO_KEYWORDS.union(PREDECLARED).union(import_names)
    code_after_funcdefs = code
    for old_fname, new_fname in func_renames.items():
        pattern = rf'(func\s*(?:\([^\)]*\)\s*)?){re.escape(old_fname)}\b'
        code_after_funcdefs = re.sub(pattern, rf'\1{new_fname}', code_after_funcdefs)

    out = []
    i = 0
    ln = len(code_after_funcdefs)
    while i < ln:
        ch = code_after_funcdefs[i]
        if ch in ('"', '`', "'"):
            # preserve string/char literals as-is for now
            end = _skip_string_or_comment_forward(code_after_funcdefs, i)
            out.append(code_after_funcdefs[i:end])
            i = end
            continue
        if code_after_funcdefs.startswith('//', i) or code_after_funcdefs.startswith('/*', i):
            end = _skip_string_or_comment_forward(code_after_funcdefs, i)
            out.append(code_after_funcdefs[i:end])
            i = end
            continue
        if IDENT_START.match(ch):
            m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', code_after_funcdefs[i:])
            ident = m.group(0)
            start = i
            end = i + m.end()
            prev_char = code_after_funcdefs[start-1] if start > 0 else ''
            replaced = ident
            if ident in func_renames and prev_char != '.':
                replaced = func_renames[ident]
            else:
                func_idx = find_innermost_function(funcs, start)
                if func_idx is not None:
                    mapping = mappings[func_idx]
                    if ident in mapping and ident not in reserved and prev_char != '.':
                        replaced = mapping[ident]
            out.append(replaced)
            i = end
            continue
        out.append(ch)
        i += 1

    obf_code = ''.join(out)

    # Encrypt sensitive strings

    # --- APPLY NEW LAYERS HERE (post-processing) ---
    obf_code = insert_random_comments(obf_code)
    obf_code = insert_dummy_functions(obf_code)
    obf_code = randomize_whitespace(obf_code)

    return obf_code, {
        'imports': sorted(list(import_names)),
        'functions_found': len(funcs),
        'signature': calculate_signature(obf_code)
    }

def transform_go(input_file, output_file, morph_interval=30):
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)
    with open(input_file, 'r', encoding='utf-8') as f:
        original_code = f.read()

    # initial obfuscation + write
    obf_code, info = obfuscate_code(original_code)
    header = f"// Auto-obfuscated by go_obfuscator.py\n// Signature: {info['signature']}\n// Imports detected (protected): {', '.join(info['imports'])}\n\n"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header + obf_code)
    print(f"[+] Obfuscated written to {output_file} (signature {info['signature']}).")

    # start a background thread to periodically re-morph the file
    def self_modify_loop():
        while True:
            time.sleep(morph_interval)
            try:
                # re-obfuscate from the original source so renames stay consistent per morph run
                new_obf, info2 = obfuscate_code(original_code)
                header2 = f"// Auto-obfuscated by go_obfuscator.py\n// Signature: {info2['signature']}\n// Imports detected (protected): {', '.join(info2['imports'])}\n\n"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(header2 + new_obf)
                print(f"[~] File morphed at {time.strftime('%Y-%m-%d %H:%M:%S')} (signature {info2['signature']}).")
            except Exception as e:
                print("[!] Morphing error:", e)
    t = threading.Thread(target=self_modify_loop, daemon=True)
    t.start()

    # attempt to run the obfuscated Go program once (no auto-restart)
    try:
        proc = subprocess.Popen(["go", "run", output_file])
        print(f"[+] Program running with PID {proc.pid}.")
    except Exception as e:
        print(f"[!] Failed to launch obfuscated program: {e}")

    # keep the script alive to continue morphing in background
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[*] Interrupted by user. Exiting.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python go_obfuscator.py input.go output.go [morph_interval_seconds]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    morph_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    transform_go(input_file, output_file, morph_interval)

if __name__ == '__main__':
    main()
