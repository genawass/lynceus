import os
from ..bundle import preflight,BundleError
from ..policy import load_ontology
from ..vocabulary import check_adapter,load_vocabulary,resolve_prompts

REQUIRED_DEPENDENCIES=('torch','transformers')

class Owlv2Adapter:
    """Local OWLv2 detector. Loads the checkpoint once and reuses it for every view."""

    def __init__(self,bundle,device=None,threshold=0.1,ontology_id=None,vocabulary_id=None):
        self.bundle=preflight(bundle)
        declared=self.bundle['manifest'].get('dependencies',{})
        missing=[p for p in REQUIRED_DEPENDENCIES if p not in declared]
        # preflight only verifies what the manifest declares, so the manifest must declare
        # everything this adapter imports at inference time.
        if missing:raise BundleError('undeclared_dependency: '+','.join(missing))
        self.threshold=threshold
        self.ontology_id=ontology_id
        self.vocabulary_id=vocabulary_id
        self._requested_device=device
        self._model=None

    def _ensure(self):
        if self._model is not None:return
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        import torch
        from transformers import Owlv2Processor,Owlv2ForObjectDetection
        self.device=self._requested_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        local=self.bundle['model_dir']
        self._processor=Owlv2Processor.from_pretrained(local,local_files_only=True)
        self._model=Owlv2ForObjectDetection.from_pretrained(local,local_files_only=True).eval().to(self.device)
        ontology=load_ontology(self.ontology_id)
        # The phrasing is configuration, not the ontology's spelling. Resolving it here means the
        # run records which vocabulary asked the question, and which classes fell back to their
        # canonical name because no vocabulary covered them.
        self._resolution=check_adapter(resolve_prompts(ontology,load_vocabulary(self.vocabulary_id)),'owlv2')
        index={c['id']:c for c in ontology['classes']}
        self._labels=[index[i] for i in self._resolution['classes']]
        self._prompts=[list(self._resolution['prompts'])]

    def capabilities(self):
        self._ensure()
        return {'device':self.device,'prompts':len(self._labels),'threshold':self.threshold,
                'vocabulary':self._resolution['vocabulary'],
                'prompt_fallbacks':self._resolution['fallbacks'],
                'prompt_coverage':self._resolution['coverage'],
                'checkpoint':self.bundle['manifest'].get('checkpoint'),'coordinate_convention':'source_pixel_xyxy',
                'ontology':load_ontology(self.ontology_id)['id']}

    def predict(self,image):
        """Detect on one view. Coordinates are in the pixel frame of the image passed in.

        Each record carries the score of every prompted class, not only the winner, because
        conservative naming has to see how close the runner-up was before it can decide whether a
        specific label is supported or only a broader one is.
        """
        self._ensure()
        import torch
        from transformers.image_transforms import center_to_corners_format
        inputs=self._processor(text=self._prompts,images=image,return_tensors='pt').to(self.device)
        with torch.inference_mode(): output=self._model(**inputs)
        # Owlv2ImageProcessor.pad squares the view with bottom/right padding of side max(h,w),
        # so predicted boxes are normalized to that square, not to the view. Unscale with the
        # padded side first, then clip the padding area away; unscaling with the view's own
        # height/width compresses every box along the shorter axis.
        side=max(image.height,image.width)
        probabilities=torch.sigmoid(output.logits[0])          # [patches, prompts]
        scores,winners=probabilities.max(dim=-1)
        keep=scores>=self.threshold
        boxes=center_to_corners_format(output.pred_boxes[0][keep])*side
        records=[]
        for box,score,winner,row in zip(boxes.tolist(),scores[keep].tolist(),
                                        winners[keep].tolist(),probabilities[keep].tolist()):
            clipped=[max(0,min(image.width,box[0])),max(0,min(image.height,box[1])),
                     max(0,min(image.width,box[2])),max(0,min(image.height,box[3]))]
            records.append({'bbox_xyxy':clipped,'raw_bbox_xyxy':box,'padded_side':side,
                            'label':self._labels[winner]['id'],'model_score':score,
                            'class_scores':{self._labels[i]['id']:s for i,s in enumerate(row)},
                            'crop_edge':box!=clipped})
        return records
