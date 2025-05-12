import copy
from ikomia import core, dataprocess, utils
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------


class InferDepthAnythingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "LiheYoung/depth-anything-base-hf"
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        self.model_name = str(params["model_name"])
        self.cuda = utils.strtobool(params["cuda"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["model_name"] = str(self.model_name)
        params["cuda"] = str(self.cuda)

        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDepthAnything(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Create parameters object
        if param is None:
            self.set_param_object(InferDepthAnythingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weights")
        self.image_processor = None
        self.model = None
        self.device = torch.device("cpu")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def load_model(self, param):
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                param.model_name,
                cache_dir=self.model_folder,
                local_files_only=True)

        except Exception as e:
            print(
                f"Failed with error: {e}. Trying without the local_files_only parameter...")
            self.image_processor = AutoImageProcessor.from_pretrained(
                param.model_name,
                cache_dir=self.model_folder)

        self.model = AutoModelForDepthEstimation.from_pretrained(
            param.model_name,
            cache_dir=self.model_folder).to(self.device)

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        h, w = src_image.shape[:2]

        # Load model
        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            self.load_model(param)
            param.update = False

        # prepare image for the model
        inputs = self.image_processor(
            images=src_image, return_tensors="pt").to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Convert depth map to RGB
        depth = F.interpolate(
            predicted_depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        # Get output (image)
        output_img = self.get_output(0)
        output_img.set_image(depth_color_rgb)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDepthAnythingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_depth_anything"
        self.info.short_description = "Depth Anything is a highly practical solution for robust monocular depth estimation"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Depth"
        self.info.version = "1.0.1"
        self.info.icon_path = "images/depth_map.jpg"
        self.info.authors = "Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang"
        self.info.article = "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"
        self.info.journal = "arXiv:2401.10891"
        self.info.year = 2024
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2401.10891"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_depth_anything"
        self.info.original_repository = "https://github.com/LiheYoung/Depth-Anything"
        # Python version
        self.info.min_python_version = "3.10.0"
        # Keywords used for search
        self.info.keywords = "Depth Estimation, Pytorch, HuggingFace, map"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OTHER"

    def create(self, param=None):
        # Create algorithm object
        return InferDepthAnything(self.info.name, param)
