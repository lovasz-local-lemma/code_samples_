"""Check that two C++ files differ only in comments (and whitespace).

Usage: verify_comment_only.py <original> <cleaned>
Exit 0 when the comment-stripped, whitespace-normalised token streams are identical.
Strings and character literals are preserved verbatim, so a comment marker inside a
string literal is not treated as a comment.
"""
import re, sys, io

def strip_comments(src: str) -> str:
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = src.find('\n', i)
            if j < 0: j = n
            out.append(' '); i = j
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            j = src.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(' '); i = j
        elif c == '"' or c == "'":
            q = c; j = i + 1
            while j < n:
                if src[j] == '\\': j += 2; continue
                if src[j] == q: break
                if src[j] == '\n': break
                j += 1
            out.append(src[i:j + 1]); i = j + 1
        else:
            out.append(c); i += 1
    return ''.join(out)

TOKEN = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|[A-Za-z_]\w*|\d[\w.]*|[^\s\w]')

def normalise(src: str) -> str:
    """Comment-stripped token stream, so whitespace differences do not count."""
    code = strip_comments(src.replace('\r\n', '\n'))
    return ' '.join(TOKEN.findall(code))

def main():
    a = io.open(sys.argv[1], encoding='utf-8', errors='replace').read()
    b = io.open(sys.argv[2], encoding='utf-8', errors='replace').read()
    na, nb = normalise(a), normalise(b)
    if na == nb:
        print(f'OK: code identical; {a.count(chr(10))} -> {b.count(chr(10))} lines')
        return 0
    # locate first divergence for the report
    k = next((i for i, (x, y) in enumerate(zip(na, nb)) if x != y), min(len(na), len(nb)))
    print('MISMATCH at normalised offset', k)
    print('  original:', na[max(0, k - 80):k + 80])
    print('  cleaned :', nb[max(0, k - 80):k + 80])
    return 1

if __name__ == '__main__':
    sys.exit(main())
