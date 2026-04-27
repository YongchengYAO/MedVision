# NOTE:
# Prompts in this file are used not only for SFT training but also model evaluation in the MedVision benchmark.
# Therefore, any modification to the prompts in this file may affect both SFT training and evaluation. 
# Please proceed with caution when editing this file.
#
# Caveat: 
# Some prompts (e.g., FORMAT_PROMPT_TL_REASONING and FORMAT_PROMPT_TUMOR_LESION_SIZE) are similar but used in different vanues. 
# Do not remove any prompt just because it looks similar to another one.

# ======================================================================================================
# Prompts used in non-CoT benchmarking
# ======================================================================================================
GENERAL_FORMAT_PROMPT = (
    "The reasoning process and the final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. "
    "For example: <think> reasoning process here </think> <answer> answer here </answer>. "
)

SYSTEM_PROMPT_LITE = (
    "A conversation between a User and an Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks through the reasoning process internally, then provides the User with the answer. "
    f"{GENERAL_FORMAT_PROMPT}"
)

FORMAT_PROMPT_BOX_COORDINATES = (
    f"{GENERAL_FORMAT_PROMPT}"
    "The answer should be four decimal numbers separated by commas without any units or additional text. "
    "The first two numbers are the coordinates of the lower-left corner and the last two numbers are the coordinates of the upper-right corner of the bounding box. "
    "Use relative coordinates in the image space, where the origin is at the lower-left corner of the image. "
    "Relative coordinates should be values between 0 and 1, representing the relative positions in the image."
)

FORMAT_PROMPT_MASK_SIZE = (
    f"{GENERAL_FORMAT_PROMPT}"
    "The answer should be a single decimal number."
)

FORMAT_PROMPT_TUMOR_LESION_SIZE = (
    f"{GENERAL_FORMAT_PROMPT}"
    "The answer should be two decimal numbers separated by a comma. "
    "The first is the major axis length, and the second is the minor axis length."
)

FORMAT_PROMPT_BIOMETRICS = (
    f"{GENERAL_FORMAT_PROMPT}"
    "The answer should be a single decimal number."
)

FORMAT_PROMPT_1_DECIMAL_NUMBER = (
    f"{GENERAL_FORMAT_PROMPT}"
    "The answer should be a single decimal number."
)
# ======================================================================================================


# ======================================================================================================
# Prompts used in SFT on datat with Chain-of-Thought (CoT) reasoning
# ======================================================================================================
FORMAT_PROMPT_TL_REASONING = (
    "The final answer must be enclosed within <answer> </answer> tags. "
    "The answer should consist of two decimal numbers separated by a comma, without units or extra text. "
    "The first number is the major axis length, and the second is the minor axis length."
)

COT_TEMPLATE_TL = (
    "<think> "
    "<step-1-reasoning> "
    "I need to identify the major axis of the ellipse enclosing the <label> and output its two endpoints. "
    "The coordinates must be written as (x1_major, y1_major), (x2_major, y2_major), where x is the width index and y is the height index. "
    "</step-1-reasoning> "
    "<step-1-answer> "
    "The endpoints of the major axis: (<x1_major>, <y1_major>), (<x2_major>, <y2_major>). "
    "</step-1-answer> "
    "<step-2-reasoning> "
    "Next, I must identify the minor axis of the ellipse enclosing the <label> and output its two endpoints in the same format: (x1_minor, y1_minor), (x2_minor, y2_minor). "
    "</step-2-reasoning> "
    "<step-2-answer> "
    "The endpoints of the minor axis: (<x1_minor>, <y1_minor>), (<x2_minor>, <y2_minor>). "
    "</step-2-answer> "
    "<step-3-reasoning> "
    "I now calculate the major axis length using the pixel dimensions (pixel_width, pixel_height) = (<pixel_width>, <pixel_height>) and the distance formula: "
    "major_axis_length = sqrt(((x2_major - x1_major) * pixel_width)^2 + ((y2_major - y1_major) * pixel_height)^2) = sqrt(((<x2_major> - <x1_major>) * <pixel_width>)^2 + ((<y2_major> - <y1_major>) * <pixel_height>)^2) = <major_axis_length>. "
    "</step-3-reasoning> "
    "<step-3-answer> "
    "The major axis length: <major_axis_length>. "
    "</step-3-answer> "
    "<step-4-reasoning> "
    "I calculate the minor axis length using the same distance formula: "
    "minor_axis_length = sqrt(((x2_minor - x1_minor) * pixel_width)^2 + ((y2_minor - y1_minor) * pixel_height)^2) = sqrt(((<x2_minor> - <x1_minor>) * <pixel_width>)^2 + ((<y2_minor> - <y1_minor>) * <pixel_height>)^2) = <minor_axis_length>. "
    "</step-4-reasoning> "
    "<step-4-answer> "
    "The minor axis length: <minor_axis_length>. "
    "</step-4-answer> "
    "</think> "
    "<answer> "
    "(<major_axis_length>, <minor_axis_length>) "
    "</answer>"
)

COT_TEMPLATE_TL_NORM = (
    "<think> "
    "<step-1-reasoning> "
    "I need to identify the major axis of the ellipse enclosing the <label> and output its two endpoints. "
    "The relative coordinates must be written as (x1_major, y1_major), (x2_major, y2_major), where x is the relative position in width and y is the relative position in height. "
    "</step-1-reasoning> "
    "<step-1-answer> "
    "The endpoints of the major axis: (<x1_major>, <y1_major>), (<x2_major>, <y2_major>). "
    "</step-1-answer> "
    "<step-2-reasoning> "
    "Next, I must identify the minor axis of the ellipse enclosing the <label> and output its two endpoints in the same format: (x1_minor, y1_minor), (x2_minor, y2_minor). "
    "</step-2-reasoning> "
    "<step-2-answer> "
    "The endpoints of the minor axis: (<x1_minor>, <y1_minor>), (<x2_minor>, <y2_minor>). "
    "</step-2-answer> "
    "<step-3-reasoning> "
    "I now calculate the major axis length using the pixel dimensions (pixel_width, pixel_height) = (<pixel_width>, <pixel_height>), the image size (image_width, image_height) = (<image_width>, <image_height>), and the distance formula: "
    "major_axis_length = sqrt(((x2_major - x1_major) * image_width * pixel_width)^2 + ((y2_major - y1_major) * image_height * pixel_height)^2) = sqrt(((<x2_major> - <x1_major>) * <image_width> * <pixel_width>)^2 + ((<y2_major> - <y1_major>) * <image_height> * <pixel_height>)^2) = <major_axis_length>. "
    "</step-3-reasoning> "
    "<step-3-answer> "
    "The major axis length: <major_axis_length>. "
    "</step-3-answer> "
    "<step-4-reasoning> "
    "I calculate the minor axis length using the same distance formula: "
    "minor_axis_length = sqrt(((x2_minor - x1_minor) * image_width * pixel_width)^2 + ((y2_minor - y1_minor) * image_height * pixel_height)^2) = sqrt(((<x2_minor> - <x1_minor>) * <image_width> * <pixel_width>)^2 + ((<y2_minor> - <y1_minor>) * <image_height> * <pixel_height>)^2) = <minor_axis_length>. "
    "</step-4-reasoning> "
    "<step-4-answer> "
    "The minor axis length: <minor_axis_length>. "
    "</step-4-answer> "
    "</think> "
    "<answer> "
    "(<major_axis_length>, <minor_axis_length>) "
    "</answer>"
)

COT_INSTRUCT_TL = (
    "Step 1: Identify the major axis (the longest diameter) of the ellipse enclosing the target region. "
    "Find its two endpoints and record their coordinates in the format (x, y) = (width index, height index). "
    "Denote the endpoints as (x1_major, y1_major) and (x2_major, y2_major). "
    "Step 2: Identify the minor axis (the shortest diameter) of the ellipse. "
    "Find its two endpoints and record their coordinates in the same (x, y) format. "
    "Denote them as (x1_minor, y1_minor) and (x2_minor, y2_minor). "
    "Step 3: Given the pixel dimensions (pixel_width, pixel_height), compute the physical length of the major axis using: "
    "major_axis_length = sqrt(((x2_major - x1_major) * pixel_width)^2 + ((y2_major - y1_major) * pixel_height)^2). "
    "Step 4: Similarly, compute the physical length of the minor axis using: "
    "minor_axis_length = sqrt(((x2_minor - x1_minor) * pixel_width)^2 + ((y2_minor - y1_minor) * pixel_height)^2). "
    "Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags, respectively. "
    "Inside <think> </think>, include reasoning and step results using "
    "<step-k-reasoning> </step-k-reasoning> and <step-k-answer> </step-k-answer> tags. "
)

COT_INSTRUCT_TL_NORM = (
    "Step 1: Identify the major axis (the longest diameter) of the ellipse enclosing the target region. "
    "Find its two endpoints and record their relative coordinates in the format (x, y) = (relative position in width direction, relative position in height direction). "
    "Denote the endpoints as (x1_major, y1_major) and (x2_major, y2_major). "
    "Step 2: Identify the minor axis (the shortest diameter) of the ellipse. "
    "Find its two endpoints and record their relative coordinates in the same (x, y) format. "
    "Denote them as (x1_minor, y1_minor) and (x2_minor, y2_minor). "
    "Step 3: Given the pixel dimensions (pixel_width, pixel_height) and image size (image_width, image_height), compute the physical length of the major axis using: "
    "major_axis_length = sqrt(((x2_major - x1_major) * image_width * pixel_width)^2 + ((y2_major - y1_major) * image_height * pixel_height)^2). "
    "Step 4: Similarly, compute the physical length of the minor axis using: "
    "minor_axis_length = sqrt(((x2_minor - x1_minor) * image_width * pixel_width)^2 + ((y2_minor - y1_minor) * image_height * pixel_height)^2). "
    "Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags, respectively. "
    "Inside <think> </think>, include reasoning and step results using "
    "<step-k-reasoning> </step-k-reasoning> and <step-k-answer> </step-k-answer> tags. "
)

FORMAT_PROMPT_AD_REASONING = (
    "The final answer must be enclosed within <answer> </answer> tags. "
    "The answer should be a single decimal number without units or extra text."
)

COT_INSTRUCT_DISTANCE = (
    "Step 1: Identify the landmark 1 and record its relative coordinates in the format (x, y) = (relative position in width direction, relative position in height direction). Denote the coordinates as (x1, y1). "
    "Step 2: Identify the landmark 2 and record its relative coordinates in the same (x, y) format. Denote the coordinates as (x2, y2). "
    "Step 3: Given the pixel dimensions (pixel_width, pixel_height) and image size (image_width, image_height), compute the physical distance between the two landmarks using: "
    "distance = sqrt(((x2 - x1) * image_width * pixel_width)^2 + ((y2 - y1) * image_height * pixel_height)^2). "
    "Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags, respectively. "
    "Inside <think> </think>, include reasoning and step results using "
    "<step-k-reasoning> </step-k-reasoning> and <step-k-answer> </step-k-answer> tags. "
)

COT_TEMPLATE_DISTANCE = (
    "<think> "
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
    "I now calculate the distance between the two landmarks using the pixel dimensions (pixel_width, pixel_height) = (<pixel_width>, <pixel_height>), the image size (image_width, image_height) = (<image_width>, <image_height>), and the distance formula: "
    "distance = sqrt(((x2 - x1) * image_width * pixel_width)^2 + ((y2 - y1) * image_height * pixel_height)^2) = sqrt(((<x2> - <x1>) * <image_width> * <pixel_width>)^2 + ((<y2> - <y1>) * <image_height> * <pixel_height>)^2) = <distance>. "
    "</step-3-reasoning> "
    "<step-3-answer> "
    "The distance: <distance>. "
    "</step-3-answer> "
    "</think> "
    "<answer> "
    "<distance> "
    "</answer>"
)

COT_INSTRUCT_ANGLE = (
    "Step 1: Identify line 1 and record the relative coordinates of its two endpoints in the format (x, y) = (relative position in width direction, relative position in height direction). Denote the endpoints as (x1_line1, y1_line1) and (x2_line1, y2_line1). "
    "Step 2: Identify line 2 and record the relative coordinates of its two endpoints in the same (x, y) format. Denote them as (x1_line2, y1_line2) and (x2_line2, y2_line2). "
    "Step 3: Given the pixel dimensions (pixel_width, pixel_height) and image size (image_width, image_height), compute the angle between the two lines using the formula: "
    "angle = arccos(|A · B| / (||A|| ||B||), where A and B are the vectors of the two lines computed from the physical coordinates of their endpoints. "
    "A = ((x2_line1 - x1_line1) * image_width * pixel_width, (y2_line1 - y1_line1) * image_height * pixel_height) and B = ((x2_line2 - x1_line2) * image_width * pixel_width, (y2_line2 - y1_line2) * image_height * pixel_height). "
    "Denote A=(Ax, Ay) and B=(Bx, By). Then, angle = arccos(|Ax*Bx + Ay*By| / (sqrt(Ax^2 + Ay^2) * sqrt(Bx^2 + By^2))). "
    "Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags, respectively. "
    "Inside <think> </think>, include reasoning and step results using "
    "<step-k-reasoning> </step-k-reasoning> and <step-k-answer> </step-k-answer> tags. "
)

COT_TEMPLATE_ANGLE = (
    "<think> "
    "<step-1-reasoning> "
    "I need to identify the relative coordinates of <landmark 1> and <landmark 2> that define line 1. "
    "The relative coordinates must be written as (x1_line1, y1_line1), (x2_line1, y2_line1), where x is the relative position in width and y is the relative position in height. "
    "</step-1-reasoning> "
    "<step-1-answer> "
    "The relative coordinates of <landmark 1> and <landmark 2>: (<x1_line1>, <y1_line1>), (<x2_line1>, <y2_line1>). "
    "</step-1-answer> "
    "<step-2-reasoning> "
    "Next, I must identify the relative coordinates of <landmark 3> and <landmark 4> that define line 2, in the same format: (x1_line2, y1_line2), (x2_line2, y2_line2). "
    "</step-2-reasoning> "
    "<step-2-answer> "
    "The relative coordinates of <landmark 3> and <landmark 4>: (<x1_line2>, <y1_line2>), (<x2_line2>, <y2_line2>). "
    "</step-2-answer> "
    "<step-3-reasoning> "
    "I now calculate the angle between the two lines using the pixel dimensions (pixel_width, pixel_height) = (<pixel_width>, <pixel_height>), the image size (image_width, image_height) = (<image_width>, <image_height>), and the angle formula: "
    "angle = arccos(|A · B| / (||A|| ||B||), where A and B are the vectors of the two lines computed from the physical coordinates of their endpoints. "
    "A = ((x2_line1 - x1_line1) * image_width * pixel_width, (y2_line1 - y1_line1) * image_height * pixel_height) = ( (<x2_line1> - <x1_line1>) * <image_width> * <pixel_width>, (<y2_line1> - <y1_line1>) * <image_height> * <pixel_height>) = (<Ax>, <Ay>). "
    "B = ((x2_line2 - x1_line2) * image_width * pixel_width, (y2_line2 - y1_line2) * image_height * pixel_height) = ( (<x2_line2> - <x1_line2>) * <image_width> * <pixel_width>, (<y2_line2> - <y1_line2>) * <image_height> * <pixel_height>) = (<Bx>, <By>). "
    "Denote A=(Ax, Ay) and B=(Bx, By). Then, angle = arccos(|Ax*Bx + Ay*By| / (sqrt(Ax^2 + Ay^2) * sqrt(Bx^2 + By^2))) = arccos(|<Ax>*<Bx> + <Ay>*<By>| / (sqrt(<Ax>^2 + <Ay>^2) * sqrt(<Bx>^2 + <By>^2))) = <angle> = <angle_degree> degrees. "
    "</step-3-reasoning> "
    "<step-3-answer> "
    "The angle: <angle_degree>. "
    "</step-3-answer> "
    "</think> "
    "<answer> "
    "<angle_degree> "
    "</answer>"
)

FORMAT_PROMPT_DETECTION_REASONING = (
    "The final answer must be enclosed within <answer> </answer> tags. "
    "The answer should be four decimal numbers separated by commas without any units or additional text. "
    "The first two numbers are the coordinates of the lower-left corner and the last two numbers are the coordinates of the upper-right corner of the bounding box. "
    "Use relative coordinates in the image space, where the origin is at the lower-left corner of the image. "
    "Relative coordinates should be values between 0 and 1, representing the relative positions in the image."
)

COT_INSTRUCT_DETECTION = (
    "Step 1: Identify the relative coordinates of the bounding box. "
    "The relative coordinates must be written as (x, y), where x is the relative position in width and y is the relative position in height. "
    "Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags, respectively. "
    "Inside <think> </think>, include reasoning and step results using "
    "<step-k-reasoning> </step-k-reasoning> and <step-k-answer> </step-k-answer> tags. "
)

COT_TEMPLATE_DETECTION = (
    "<think> "
    "<step-1-reasoning> "
    "I need to identify the relative coordinates of the bounding box of <label_name>. "
    "The relative coordinates must be written as (x, y), where x is the relative position in width and y is the relative position in height. "
    "I understand that the origin of the image space coordinate system is at the lower-left corner of the image. "
    "</step-1-reasoning> "
    "<step-1-answer> "
    "The relative coordinates of the bounding box: (<coor0_w>, <coor0_h>), (<coor1_w>, <coor1_h>). "
    "</step-1-answer> "
    "</think> "
    "<answer> "
    "<coor0_w>,<coor0_h>,<coor1_w>,<coor1_h>"
    "</answer>"
)
# ======================================================================================================


def fill_in_template(template, values_dict):
    filled_template = template
    for key, value in values_dict.items():
        filled_template = filled_template.replace(key, str(value))
    return filled_template


def _get_prompt_angle(biometrics_name, l1p1, l1p2, l2p1, l2p2, metric_unit):
    """Prepare prompt for angle estimate VQA. Inputs are names."""
    if biometrics_name is not None and biometrics_name != "":
        return (
            f"estimate the angle of {biometrics_name} in {metric_unit}, "
            f"which is the angle between 2 lines: "
            f"(line 1) the line connecting {l1p1} and {l1p2}, "
            f"(line 2) the line connecting {l2p1} and {l2p2}.\n"
        )
    else:
        return (
            f"estimate the angle between 2 lines in {metric_unit}: "
            f"(line 1) the line connecting {l1p1} and {l1p2}, "
            f"(line 2) the line connecting {l2p1} and {l2p2}.\n"
        )


def _get_prompt_distance(biometrics_name, p1, p2, metric_unit):
    """Prepare prompt for distance estimate VQA. Inputs are names."""
    if biometrics_name is not None and biometrics_name != "":
        return (
            f"estimate the distance of {biometrics_name} in {metric_unit}, "
            f"which is the distance between 2 landmark points: "
            f"(landmark 1) {p1}, "
            f"(landmark 2) {p2}.\n"
        )
    else:
        return (
            f"estimate the distance between 2 landmark points in {metric_unit}: "
            f"(landmark 1) {p1}, "
            f"(landmark 2) {p2}.\n"
        )
