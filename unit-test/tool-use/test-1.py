print("=== Test 1: safe_exec_python sandbox ===")
print("Objective : Verify safe_exec_python correctly executes the three tool-use code templates")
print("            (distance, angle, TL) and blocks dangerous operations.")
print("Expected  :")
print("  distance template -> correct numeric result matching Python reference")
print("  angle template    -> correct numeric result matching Python reference")
print("  TL template       -> correct 'major, minor' string matching Python reference")
print("  open() call       -> ERROR: (dangerous builtin blocked by sandbox)")
print("  raised exception  -> ERROR: (exception caught and returned as string)")
import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path("src").resolve()))

from medvision_bm.utils.tool_execution import safe_exec_python

# --- distance template ---
code = (
    "import math\n"
    "x1,y1=0.1,0.2\n"
    "x2,y2=0.5,0.8\n"
    "W,H=512,512\n"
    "pw,ph=0.5,0.5\n"
    "print(round(math.sqrt(((x2-x1)*W*pw)**2+((y2-y1)*H*ph)**2),3))"
)
result = safe_exec_python(code)
expected = str(round(math.sqrt(((0.5-0.1)*512*0.5)**2 + ((0.8-0.2)*512*0.5)**2), 3))
assert result == expected, f"distance: got {result!r}, expected {expected!r}"
print(f"  distance template    : {result}  (expected {expected})  PASS")

# --- angle template ---
code = (
    "import math\n"
    "Ax=(0.5-0.1)*512*0.5; Ay=(0.8-0.2)*512*0.5\n"
    "Bx=(0.9-0.3)*512*0.5; By=(0.7-0.4)*512*0.5\n"
    "cos_t=abs(Ax*Bx+Ay*By)/(math.sqrt(Ax**2+Ay**2)*math.sqrt(Bx**2+By**2))\n"
    "print(round(math.degrees(math.acos(min(cos_t,1.0))),3))"
)
result = safe_exec_python(code)
Ax = (0.5-0.1)*512*0.5; Ay = (0.8-0.2)*512*0.5
Bx = (0.9-0.3)*512*0.5; By = (0.7-0.4)*512*0.5
cos_t = abs(Ax*Bx+Ay*By) / (math.sqrt(Ax**2+Ay**2) * math.sqrt(Bx**2+By**2))
expected = str(round(math.degrees(math.acos(min(cos_t, 1.0))), 3))
assert result == expected, f"angle: got {result!r}, expected {expected!r}"
print(f"  angle template       : {result}  (expected {expected})  PASS")

# --- TL template (major + minor axis lengths) ---
code = (
    "import math\n"
    "major=math.sqrt((((0.5-0.1)*512*0.5)**2+((0.8-0.2)*512*0.5)**2))\n"
    "minor=math.sqrt((((0.7-0.3)*512*0.5)**2+((0.6-0.4)*512*0.5)**2))\n"
    "print(f'{round(major,3)}, {round(minor,3)}')"
)
result = safe_exec_python(code)
exp_major = round(math.sqrt(((0.5-0.1)*512*0.5)**2 + ((0.8-0.2)*512*0.5)**2), 3)
exp_minor = round(math.sqrt(((0.7-0.3)*512*0.5)**2 + ((0.6-0.4)*512*0.5)**2), 3)
expected = f"{exp_major}, {exp_minor}"
assert result == expected, f"TL: got {result!r}, expected {expected!r}"
print(f"  TL template          : {result!r}  PASS")

# --- blocked builtin: open() must be denied ---
result = safe_exec_python("open('/etc/passwd', 'r')")
assert result.startswith("ERROR:"), f"expected ERROR:, got {result!r}"
print(f"  blocked open()       : {result[:50]!r}  PASS")

# --- exception inside sandbox is caught ---
result = safe_exec_python("raise ValueError('test error')")
assert result.startswith("ERROR:"), f"expected ERROR:, got {result!r}"
print(f"  raised exception     : {result[:50]!r}  PASS")

print("OK")
