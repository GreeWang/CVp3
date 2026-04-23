import re

with open("src/cv_project/inpainting/diffusion_enhancer.py", "r") as f:
    content = f.read()

# 1. remove staticmethod decorator and add self argument to _warp_tensor_to_target
content = content.replace("@staticmethod\n    def _warp_tensor_to_target(\n", "    def _warp_tensor_to_target(\n        self,\n")\
                 .replace("@staticmethod\n    def _warp_tensor_to_target(source_frame:", "    def _warp_tensor_to_target(self, source_frame:")

# 2. replace the inner OpenCV flow call with dynamic self._compute_flow
old_flow_code = """        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            target_gray,
            source_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        height, width = target_gray.shape"""

new_flow_code = """        flow = self._compute_flow(source_frame, target_frame)
        height, width = flow.shape[:2]"""

content = content.replace(old_flow_code, new_flow_code)

# 3. Add the helper methods at the end of the class
methods_to_add = """
    def _lazy_load_raft(self):
        if hasattr(self, "_raft_model"):
            return self._raft_model, getattr(self, "_raft_device", "cpu")

        import sys
        import os
        import torch

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        propainter_dir = self.config.get("propainter_repo_dir", "ProPainter")
        raft_dir = os.path.join(project_root, propainter_dir)

        if raft_dir not in sys.path:
            sys.path.insert(0, raft_dir)

        try:
            from model.modules.flow_comp_raft import initialize_RAFT
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_path = os.path.join(raft_dir, "weights", "raft-things.pth")

            raft_model = initialize_RAFT(model_path, device=device)
            raft_model.eval()
            self._raft_model = raft_model
            self._raft_device = device
            return raft_model, device
        except Exception as e:
            print(f"Failed to load RAFT: {e}")
            self._raft_model = None
            self._raft_device = None
            return None, None

    def _compute_flow(self, source_frame: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
        raft_model, device = self._lazy_load_raft()
        if raft_model is not None:
            import torch
            import cv2
            
            t_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
            s_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)

            h, w = t_rgb.shape[:2]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            if pad_h > 0 or pad_w > 0:
                t_rgb = cv2.copyMakeBorder(t_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                s_rgb = cv2.copyMakeBorder(s_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

            t_tensor = (torch.from_numpy(t_rgb).permute(2, 0, 1).float() / 255.0) * 2.0 - 1.0
            s_tensor = (torch.from_numpy(s_rgb).permute(2, 0, 1).float() / 255.0) * 2.0 - 1.0

            t_tensor = t_tensor.unsqueeze(0).to(device)
            s_tensor = s_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                # target->source backward logic: feed target as image1, source as image2
                _, flows_forward = raft_model(t_tensor, s_tensor, iters=20, test_mode=True)

            flow_np = flows_forward[0].permute(1, 2, 0).cpu().numpy()
            
            if pad_h > 0 or pad_w > 0:
                flow_np = flow_np[:h, :w, :]
                
            return flow_np
        else:
            # Fallback
            import cv2
            target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
            source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
            return cv2.calcOpticalFlowFarneback(
                target_gray,
                source_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
"""

if "def _lazy_load_raft" not in content:
    content = content + methods_to_add

with open("src/cv_project/inpainting/diffusion_enhancer.py", "w") as f:
    f.write(content)

