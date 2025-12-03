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
    "<step-1-reasoning> I need to identify the major axis of the ellipse enclosing the <label> and output its two endpoints. "
    "The coordinates must be written as (x1_major, y1_major), (x2_major, y2_major), where x is the width index and y is the height index. "
    "</step-1-reasoning> "
    "<step-1-answer> The endpoints of the major axis: (<x1_major>, <y1_major>), (<x2_major>, <y2_major>). </step-1-answer> "
    "<step-2-reasoning> Next, I must identify the minor axis of the ellipse enclosing the <label> and output its two endpoints "
    "in the same format: (x1_minor, y1_minor), (x2_minor, y2_minor). </step-2-reasoning> "
    "<step-2-answer> The endpoints of the minor axis: (<x1_minor>, <y1_minor>), (<x2_minor>, <y2_minor>). </step-2-answer> "
    "<step-3-reasoning> I now calculate the major axis length using the pixel dimensions (pixel_width, pixel_height) = (<pixel_width>, <pixel_height>) and the distance formula: "
    "major_axis_length = sqrt(((x2_major - x1_major) * pixel_width)^2 + ((y2_major - y1_major) * pixel_height)^2) = sqrt(((<x2_major> - <x1_major>) * <pixel_width>)^2 + ((<y2_major> - <y1_major>) * <pixel_height>)^2) = <major_axis_length>. "
    "</step-3-reasoning> "
    "<step-3-answer> The major axis length: <major_axis_length>. </step-3-answer> "
    "<step-4-reasoning> I calculate the minor axis length using the same distance formula: "
    "minor_axis_length = sqrt(((x2_minor - x1_minor) * pixel_width)^2 + ((y2_minor - y1_minor) * pixel_height)^2) = sqrt(((<x2_minor> - <x1_minor>) * <pixel_width>)^2 + ((<y2_minor> - <y1_minor>) * <pixel_height>)^2) = <minor_axis_length>. "
    "</step-4-reasoning> "
    "<step-4-answer> The minor axis length: <minor_axis_length>. </step-4-answer> "
    "</think> "
    "<answer> (<major_axis_length>, <minor_axis_length>) </answer>"
)

COT_TEMPLATE_TL_NORM = (
    "<think> "
    "<step-1-reasoning> I need to identify the major axis of the ellipse enclosing the <label> and output its two endpoints. "
    "The relative coordinates must be written as (x1_major, y1_major), (x2_major, y2_major), where x is the relative position in width and y is the relative position in height. "
    "</step-1-reasoning> "
    "<step-1-answer> The endpoints of the major axis: (<x1_major>, <y1_major>), (<x2_major>, <y2_major>). </step-1-answer> "
    "<step-2-reasoning> Next, I must identify the minor axis of the ellipse enclosing the <label> and output its two endpoints "
    "in the same format: (x1_minor, y1_minor), (x2_minor, y2_minor). </step-2-reasoning> "
    "<step-2-answer> The endpoints of the minor axis: (<x1_minor>, <y1_minor>), (<x2_minor>, <y2_minor>). </step-2-answer> "
    "<step-3-reasoning> I now calculate the major axis length using the pixel dimensions (pixel_width, pixel_height) = (<pixel_width>, <pixel_height>), the image size (image_width, image_height) = (<image_width>, <image_height>), and the distance formula: "
    "major_axis_length = sqrt(((x2_major - x1_major) * image_width * pixel_width)^2 + ((y2_major - y1_major) * image_height * pixel_height)^2) = sqrt(((<x2_major> - <x1_major>) * <image_width> * <pixel_width>)^2 + ((<y2_major> - <y1_major>) * <image_height> * <pixel_height>)^2) = <major_axis_length>. "
    "</step-3-reasoning> "
    "<step-3-answer> The major axis length: <major_axis_length>. </step-3-answer> "
    "<step-4-reasoning> I calculate the minor axis length using the same distance formula: "
    "minor_axis_length = sqrt(((x2_minor - x1_minor) * image_width * pixel_width)^2 + ((y2_minor - y1_minor) * image_height * pixel_height)^2) = sqrt(((<x2_minor> - <x1_minor>) * <image_width> * <pixel_width>)^2 + ((<y2_minor> - <y1_minor>) * <image_height> * <pixel_height>)^2) = <minor_axis_length>. "
    "</step-4-reasoning> "
    "<step-4-answer> The minor axis length: <minor_axis_length>. </step-4-answer> "
    "</think> "
    "<answer> (<major_axis_length>, <minor_axis_length>) </answer>"
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

# # old version kept for reference
# FORMAT_PROMPT_BOX_COORDINATES_REASONING = (
#      "The answer should be four decimal numbers separated by commas without any units or additional text. "
#     "The first two numbers are the coordinates of the lower-left corner and the last two numbers are the coordinates of the upper-right corner of the bounding box. "
#     "Use relative coordinates in the image space, where the origin is at the lower-left corner of the image. "
#     "Relative coordinates should be values between 0 and 1, representing the relative positions in the image."
# )
# ======================================================================================================


# ======================================================================================================
# Prompts used in non-CoT benchmarking
# ======================================================================================================
FORMAT_PROMPT_BOX_COORDINATES = (
    "The answer should be four decimal numbers separated by commas without any units or additional text. "
    "The first two numbers are the coordinates of the lower-left corner and the last two numbers are the coordinates of the upper-right corner of the bounding box. "
    "Use relative coordinates in the image space, where the origin is at the lower-left corner of the image. "
    "Relative coordinates should be values between 0 and 1, representing the relative positions in the image."
)

FORMAT_PROMPT_MASK_SIZE = (
    "The answer should be 1 decimal number without any units or additional text."
)

FORMAT_PROMPT_TUMOR_LESION_SIZE = (
    "The answer should be two decimal numbers separated by a comma without any units or additional text. "
    "The first is the major axis length, and the second is the minor axis length."
)

FORMAT_PROMPT_BIOMETRICS = (
    "The answer should be a single decimal number without any units or additional text."
)
# ======================================================================================================


def fill_in_template(template, values_dict):
    filled_template = template
    for key, value in values_dict.items():
        filled_template = filled_template.replace(key, str(value))
    return filled_template
