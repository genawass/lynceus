"""Grounding DINO as an independent verifier.

A third lineage, which is the whole reason to add it. OWLv2 scores a CLIP-style text tower against
image patches and SAM 3 asks a presence head whether a concept is in the view; Grounding DINO
fuses a Swin backbone with a BERT encoder and grounds phrases to boxes. Two models that fail alike
confirm each other's failures, so architectural separation is the point -- though it is a matter of
degree, since all three are transformers trained on overlapping web-scale image-text data. Whether
this one actually decorrelates is measured, not assumed: its sensitivity on known positives is
observable, and so is how much its confirmations overlap another verifier's.

The label path needs care. Grounding DINO consumes one period-separated phrase list and returns
spans over that text rather than class indices, and the span it returns need not be a phrase that
was sent -- it can be a fragment, a merge of two neighbours, or a subword. So the mapping back to
ontology classes is a real step that can fail, and it fails silently unless counted. Every
detection carries the span it came from, and detections whose span matches no prompt are reported
rather than dropped quietly.
"""
import os

REQUIRED_DEPENDENCIES = ('torch', 'transformers')


def build_prompt(phrases):
    """Grounding DINO's expected input: lowercase noun phrases, period separated, trailing period."""
    return ' '.join(f'{phrase.strip().lower().rstrip(".")}.' for phrase in phrases)


def map_span(span, phrases):
    """Which prompted phrase a returned span refers to, or None when nothing matches.

    Exact match first, then containment in either direction, so `motorcycle` returned as `motor`
    and `person` returned as `a person` both resolve while an unrelated span does not. Ties go to
    the longest phrase, because a span containing `bus` and `minibus` refers to the more specific
    one that was actually prompted.
    """
    cleaned = span.strip().lower().rstrip('.')
    if not cleaned:
        return None
    candidates = [p for p in phrases if p.strip().lower().rstrip('.') == cleaned]
    if not candidates:
        candidates = [p for p in phrases
                      if cleaned in p.strip().lower() or p.strip().lower() in cleaned]
    return max(candidates, key=len) if candidates else None


class GroundingDinoVerifier:
    """Local Grounding DINO verifier. One pass per tile serves every prompted class."""

    def __init__(self, bundle, device=None, tile_side=512, threshold=0.3, text_threshold=0.25, amp=True):
        from ..bundle import preflight, BundleError
        self.bundle = preflight(bundle)
        declared = self.bundle['manifest'].get('dependencies', {})
        missing = [p for p in REQUIRED_DEPENDENCIES if p not in declared]
        if missing:
            raise BundleError('undeclared_dependency: ' + ','.join(missing))
        self.tile_side = tile_side
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.amp = amp
        self._requested_device = device
        self._model = None
        self.unmapped_spans = {}

    def _ensure(self):
        if self._model is not None:
            return
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        self.device = self._requested_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        local = self.bundle['model_dir']
        self._processor = AutoProcessor.from_pretrained(local, local_files_only=True)
        self._model = GroundingDinoForObjectDetection.from_pretrained(
            local, local_files_only=True).eval().to(self.device)

    def capabilities(self):
        self._ensure()
        return {'device': self.device, 'tile_side': self.tile_side, 'threshold': self.threshold,
                'text_threshold': self.text_threshold, 'amp': self.amp,
                'checkpoint': self.bundle['manifest'].get('checkpoint'),
                'coordinate_convention': 'source_pixel_xyxy', 'prompt': 'phrase_list',
                'label_path': 'text spans mapped back to ontology classes; unmapped spans counted',
                'independence': 'distinct backbone and text encoder from the detector; '
                                'shared web-scale pretraining corpus, so decorrelation is partial'}

    def detect(self, image, prompts):
        """Every detection in source coordinates, with spans mapped back to ontology classes."""
        self._ensure()
        import torch
        from .sam3 import plan_windows
        phrases = list(prompts)
        text = build_prompt(phrases)
        windows = plan_windows(image.width, image.height, self.tile_side)
        found = []
        for x1, y1, x2, y2 in windows:
            view = image.crop((x1, y1, x2, y2))
            inputs = self._processor(images=view, text=text, return_tensors='pt').to(self.device)
            with torch.inference_mode():
                if self.amp and self.device == 'cuda':
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = self._model(**inputs)
                else:
                    output = self._model(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                output, inputs['input_ids'], threshold=self.threshold,
                text_threshold=self.text_threshold, target_sizes=[view.size[::-1]])[0]
            for box, score, span in zip(result['boxes'].cpu().tolist(),
                                        result['scores'].cpu().tolist(),
                                        result.get('text_labels', result.get('labels'))):
                phrase = map_span(str(span), phrases)
                if phrase is None:
                    self.unmapped_spans[str(span)] = self.unmapped_spans.get(str(span), 0) + 1
                    continue
                found.append({'bbox_xyxy': [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1],
                              # The class id travels, never the phrasing.
                              'label': prompts[phrase], 'score': float(score), 'span': str(span)})
        return found
