"""WeDetect-Uni as a class-agnostic verifier.

The two verifiers tried before are prompted: they are asked whether a named concept is present,
so their answer depends on the vocabulary and on how each phrase happens to be worded. WeDetect-Uni
is not asked anything. It returns generic object proposals, which makes it the only verifier here
whose agreement cannot be confounded by naming -- and it separates on the axis that mattered least
for the last one, a ConvNeXt backbone against two vision transformers.

Its input resolution is the other reason to try it. The large variant letterboxes to 1280, and the
panel's frames are 1360x765, so a whole frame arrives at 0.94 of native scale in a single pass
where the prompted detector needs seven tiled views to get there.

The implementation lives outside this repository on purpose. WeDetect is GPL-v3, so vendoring it
would relicense this tree; the bundle names the directory to import from and records the licence,
and WP-14 can refuse to distribute a bundle carrying one. That keeps the model usable for
measurement without making a distribution decision by accident.
"""
import sys
from pathlib import Path

REQUIRED_DEPENDENCIES = ('torch',)


class WeDetectProposer:
    """Local WeDetect-Uni. Returns class-agnostic proposals, never a label."""

    def __init__(self, bundle, device=None, tile_side=None, threshold=0.2,
                 num_proposals=900, code_dir=None):
        from ..bundle import preflight, BundleError
        self.bundle = preflight(bundle)
        manifest = self.bundle['manifest']
        declared = manifest.get('dependencies', {})
        missing = [p for p in REQUIRED_DEPENDENCIES if p not in declared]
        if missing:
            raise BundleError('undeclared_dependency: ' + ','.join(missing))
        self.code_dir = code_dir or manifest.get('code_dir')
        if not self.code_dir:
            raise BundleError('missing_code_dir: WeDetect is loaded from source, not from a served model')
        self.checkpoint = Path(self.bundle['model_dir']) / manifest['checkpoint_file']
        self.backbone_size = manifest.get('backbone_size', 'large')
        # None means one pass over the whole frame, which is what the 1280 input is for.
        self.tile_side = tile_side
        self.threshold = threshold
        self.num_proposals = num_proposals
        self._requested_device = device
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        if str(self._requested_device or 'cuda') != 'cuda':
            # The upstream forward moves inputs with a hardcoded .cuda(), so anything else would
            # fail inside the model rather than here.
            raise RuntimeError('wedetect_requires_cuda')
        self.device = 'cuda'
        if self.code_dir not in sys.path:
            sys.path.insert(0, self.code_dir)
        from generate_proposal import SimpleYOLOWorldDetector
        model = SimpleYOLOWorldDetector(backbone_size=self.backbone_size, prompt_dim=768,
                                        num_prompts=256, num_proposals=self.num_proposals)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        state = _remap(state)
        report = model.load_state_dict(state, strict=False)
        self._load_report = {'missing': len(report.missing_keys), 'unexpected': len(report.unexpected_keys)}
        self._model = model.cuda().eval()

    def capabilities(self):
        self._ensure()
        return {'device': self.device, 'tile_side': self.tile_side, 'threshold': self.threshold,
                'num_proposals': self.num_proposals, 'backbone': f'ConvNeXt-{self.backbone_size}',
                'input_size': 1280 if self.backbone_size == 'large' else 640,
                'checkpoint': self.bundle['manifest'].get('checkpoint'),
                'coordinate_convention': 'source_pixel_xyxy', 'prompt': 'none_class_agnostic',
                'state_dict_load': self._load_report, 'licence': self.bundle['manifest'].get('licence'),
                'independence': 'convolutional backbone and no text conditioning; '
                                'shared web-scale pretraining corpus, so decorrelation is partial'}

    def detect(self, image, prompts=None):
        """Class-agnostic proposals in source coordinates.

        `prompts` is accepted and ignored so this satisfies the verifier interface. Every record
        carries `label: None`, because a proposal makes no claim about what the object is and
        pretending otherwise would let a naming disagreement look like a presence disagreement.
        """
        self._ensure()
        import torch
        if self.tile_side:
            from .sam3 import plan_windows
            windows = plan_windows(image.width, image.height, self.tile_side)
        else:
            windows = [[0, 0, image.width, image.height]]
        found = []
        for x1, y1, x2, y2 in windows:
            view = image if len(windows) == 1 else image.crop((x1, y1, x2, y2))
            with torch.inference_mode():
                output = self._model([view])[0]
            boxes = output['bboxes'].float().cpu().tolist()
            scores = output['scores'].float().cpu().tolist()
            for box, score in zip(boxes, scores):
                if score < self.threshold:
                    continue
                found.append({'bbox_xyxy': [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1],
                              'label': None, 'score': float(score)})
        return found


def _remap(state):
    """The released checkpoint is keyed for the training graph; the inference model renames both."""
    for key in [k for k in state if 'backbone' in k]:
        state[key.replace('backbone.image_model.model.', 'backbone.')] = state.pop(key)
    for key in [k for k in state if 'bbox_head' in k]:
        new = key.replace('bbox_head.head_module.', 'bbox_head.')
        for old_part, new_part in (('0.2.', '0.6.'), ('1.2.', '1.6.'), ('2.2.', '2.6.'),
                                   ('1.bn', '4'), ('1.conv', '3'), ('0.bn', '1'), ('0.conv', '0')):
            new = new.replace(old_part, new_part)
        state[new] = state.pop(key)
    return state
