# src/medvision_bm/sft/sft_prompts_tooluse.py

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code and return printed output",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

# Python code templates — filled with .format(**kwargs) at data-generation time.
# Template variables use single-brace Python format syntax (no angle brackets).

PYTHON_TEMPLATE_DISTANCE = (
    "import math\n"
    "x1,y1={x1},{y1}\n"
    "x2,y2={x2},{y2}\n"
    "W,H={W},{H}\n"
    "pw,ph={pw},{ph}\n"
    "print(round(math.sqrt(((x2-x1)*W*pw)**2+((y2-y1)*H*ph)**2),3))"
)

PYTHON_TEMPLATE_ANGLE = (
    "import math\n"
    "Ax=({x2}-{x1})*{W}*{pw}; Ay=({y2}-{y1})*{H}*{ph}\n"
    "Bx=({x4}-{x3})*{W}*{pw}; By=({y4}-{y3})*{H}*{ph}\n"
    "cos_t=abs(Ax*Bx+Ay*By)/(math.sqrt(Ax**2+Ay**2)*math.sqrt(Bx**2+By**2))\n"
    "print(round(math.degrees(math.acos(min(cos_t,1.0))),3))"
)

PYTHON_TEMPLATE_TL = (
    "import math\n"
    "major=math.sqrt(((({x2}-{x1})*{W}*{pw})**2+(({y2}-{y1})*{H}*{ph})**2))\n"
    "minor=math.sqrt(((({x4}-{x3})*{W}*{pw})**2+(({y4}-{y3})*{H}*{ph})**2))\n"
    "print(f'{{round(major,3)}}, {{round(minor,3)}}')"
)

# Think block text for assistant turn 3 (steps 1-2 only; step 3 triggers tool call).
# Keys match the values_dict keys from _doc_to_text_AngleDistanceTask_CoT and
# _doc_to_text_TumorLesionTask_CoT (angle-bracket style, filled by fill_in_template).

COT_THINK_DISTANCE_TOOLUSE = (
    "<step-1-reasoning> "
    "I need to identify <landmark 1> and output its relative coordinates. "
    "The relative coordinates must be written as (x1, y1), where x is the relative position in width and y is the relative position in height. "
    "</step-1-reasoning> "
    "<step-1-answer> "
    "The relative coordinates of <landmark 1>: (<x1>, <y1>). "
    "</step-1-answer> "
    "<step-2-reasoning> "
    "Next, I must identify <landmark 2> and output its relative coordinates in the same format: (x2, y2). "
    "</step-2-reasoning> "
    "<step-2-answer> "
    "The relative coordinates of <landmark 2>: (<x2>, <y2>). "
    "</step-2-answer> "
    "<step-3-reasoning> "
    "I will call execute_python to compute the distance using: "
    "distance = sqrt(((x2-x1)*W*pw)**2+((y2-y1)*H*ph)**2) "
    "with pixel dimensions (pw, ph) = (<pixel_width>, <pixel_height>) and image size (W, H) = (<image_width>, <image_height>). "
    "</step-3-reasoning>"
)

COT_THINK_ANGLE_TOOLUSE = (
    "<step-1-reasoning> "
    "I need to identify the relative coordinates of <landmark 1> and <landmark 2> that define line 1. "
    "The relative coordinates must be written as (x1_line1, y1_line1), (x2_line1, y2_line1). "
    "</step-1-reasoning> "
    "<step-1-answer> "
    "The relative coordinates of <landmark 1> and <landmark 2>: (<x1_line1>, <y1_line1>), (<x2_line1>, <y2_line1>). "
    "</step-1-answer> "
    "<step-2-reasoning> "
    "Next, I must identify the relative coordinates of <landmark 3> and <landmark 4> that define line 2. "
    "</step-2-reasoning> "
    "<step-2-answer> "
    "The relative coordinates of <landmark 3> and <landmark 4>: (<x1_line2>, <y1_line2>), (<x2_line2>, <y2_line2>). "
    "</step-2-answer> "
    "<step-3-reasoning> "
    "I will call execute_python to compute the angle using vectors A and B from physical coordinates. "
    "pixel dimensions (pw, ph) = (<pixel_width>, <pixel_height>), image size (W, H) = (<image_width>, <image_height>). "
    "</step-3-reasoning>"
)

COT_THINK_TL_TOOLUSE = (
    "<step-1-reasoning> "
    "I need to identify the major axis of the ellipse enclosing the <label> and output its two endpoints. "
    "Relative coordinates: (x1_major, y1_major), (x2_major, y2_major). "
    "</step-1-reasoning> "
    "<step-1-answer> "
    "The endpoints of the major axis: (<x1_major>, <y1_major>), (<x2_major>, <y2_major>). "
    "</step-1-answer> "
    "<step-2-reasoning> "
    "Next, I must identify the minor axis endpoints: (x1_minor, y1_minor), (x2_minor, y2_minor). "
    "</step-2-reasoning> "
    "<step-2-answer> "
    "The endpoints of the minor axis: (<x1_minor>, <y1_minor>), (<x2_minor>, <y2_minor>). "
    "</step-2-answer> "
    "<step-3-reasoning> "
    "I will call execute_python to compute both axis lengths. "
    "pixel dimensions (pw, ph) = (<pixel_width>, <pixel_height>), image size (W, H) = (<image_width>, <image_height>). "
    "</step-3-reasoning>"
)

# User prompt instruction suffix (replaces COT_INSTRUCT_* from sft_prompts.py)
COT_INSTRUCT_DISTANCE_TOOLUSE = (
    "Step 1: Identify landmark 1 and record its relative coordinates (x1, y1). "
    "Step 2: Identify landmark 2 and record its relative coordinates (x2, y2). "
    "Step 3: Call the execute_python tool with a Python script that computes the physical distance using: "
    "distance = sqrt(((x2-x1)*W*pw)**2+((y2-y1)*H*ph)**2). "
    "Step 4: Report the tool result in <answer> </answer> tags. "
    "Report your reasoning in <think> </think> tags. "
    "The final answer must be a single decimal number without units or extra text."
)

COT_INSTRUCT_ANGLE_TOOLUSE = (
    "Step 1: Identify the two endpoints of line 1 and record their relative coordinates. "
    "Step 2: Identify the two endpoints of line 2 and record their relative coordinates. "
    "Step 3: Call the execute_python tool with a Python script that computes the angle between the two lines using vectors from physical coordinates. "
    "Step 4: Report the tool result in <answer> </answer> tags. "
    "Report your reasoning in <think> </think> tags. "
    "The final answer must be a single decimal number without units or extra text."
)

COT_INSTRUCT_TL_TOOLUSE = (
    "Step 1: Identify the major axis endpoints of the ellipse and record their relative coordinates. "
    "Step 2: Identify the minor axis endpoints and record their relative coordinates. "
    "Step 3: Call the execute_python tool with a Python script that computes both axis lengths. "
    "Step 4: Report the tool results as 'major_length, minor_length' in <answer> </answer> tags. "
    "Report your reasoning in <think> </think> tags. "
    "The final answer must be two decimal numbers separated by a comma without units or extra text."
)
