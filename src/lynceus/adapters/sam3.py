"""SAM 3 boundary refinement through the visual-prompt path.

A refiner takes the boxes a detector proposed and returns better boxes for the same instances. It
adds nothing and removes nothing, so it cannot change what was discovered, only where the
boundaries sit.

Refinement runs inside overlapping source-resolution tiles, one model pass serving every box in a
tile. A whole-frame pass resizes the view to the model's input and delivers a small object at a
fraction of its pixels; a per-box crop pays a full encoder pass to upscale a window carrying no
more information than the tile already did. Measured on two panels, the tiled form is better than
the whole-frame one for every model tried.

`Sam3Model` with `input_boxes` returns DETR-style detections rather than a mask per prompt, so the
box-prompt path is `Sam3TrackerModel`. SAM 3 needs a `transformers` newer than the one this bundle
pins, which is why the import happens here at the point of use: a run without refinement neither
imports it nor requires it.
"""
import os

REQUIRED_DEPENDENCIES=('torch','transformers')


def plan_windows(width,height,side,overlap=0.5):
    step=max(1,int(side*(1-overlap)))
    xs=sorted({min(x,max(0,width-side)) for x in range(0,max(1,width),step)})
    ys=sorted({min(y,max(0,height-side)) for y in range(0,max(1,height),step)})
    return [[x,y,min(x+side,width),min(y+side,height)] for y in ys for x in xs]


def assign(box,windows):
    """The window containing the box with the most room to spare, or None if none contains it.

    Margin decides rather than mere containment: a box touching its view's border is the
    truncation case the merge rule already works around, and refinement must not manufacture more.
    """
    best,best_margin=None,-1.0
    for index,window in enumerate(windows):
        if box[0]>=window[0] and box[1]>=window[1] and box[2]<=window[2] and box[3]<=window[3]:
            margin=min(box[0]-window[0],box[1]-window[1],window[2]-box[2],window[3]-box[3])
            if margin>best_margin: best,best_margin=index,margin
    return best


def mask_to_box(mask):
    """Tight box around a mask, or None when it is empty.

    The whole mask is used rather than its largest component, because the annotation policy keeps
    disconnected visible portions of one object inside its box; dropping a component here would
    decide a policy question this step is not entitled to decide.
    """
    import numpy as np
    rows=np.any(mask,axis=1);columns=np.any(mask,axis=0)
    if not rows.any() or not columns.any():return None
    y1,y2=np.where(rows)[0][[0,-1]];x1,x2=np.where(columns)[0][[0,-1]]
    return [float(x1),float(y1),float(x2+1),float(y2+1)]


class Sam3Refiner:
    """Local SAM 3 refiner. Loads the checkpoint once and reuses it for every view."""

    def __init__(self,bundle,device=None,tile_side=512,max_prompts=64,amp=True):
        from ..bundle import preflight,BundleError
        self.bundle=preflight(bundle)
        declared=self.bundle['manifest'].get('dependencies',{})
        missing=[p for p in REQUIRED_DEPENDENCIES if p not in declared]
        if missing:raise BundleError('undeclared_dependency: '+','.join(missing))
        self.tile_side=tile_side;self.max_prompts=max_prompts;self.amp=amp
        self._requested_device=device;self._model=None

    def _ensure(self):
        if self._model is not None:return
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        import torch
        from transformers import Sam3TrackerModel,Sam3TrackerProcessor
        self.device=self._requested_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        local=self.bundle['model_dir']
        self._processor=Sam3TrackerProcessor.from_pretrained(local,local_files_only=True)
        self._model=Sam3TrackerModel.from_pretrained(local,local_files_only=True).eval().to(self.device)

    def capabilities(self):
        self._ensure()
        return {'device':self.device,'tile_side':self.tile_side,'max_prompts':self.max_prompts,
                'checkpoint':self.bundle['manifest'].get('checkpoint'),
                'coordinate_convention':'source_pixel_xyxy','prompt':'box','amp':self.amp}

    def refine(self,image,boxes):
        """One record per input box: the refined geometry and the model's own mask quality.

        A box no tile fully contains, or whose mask comes back empty, yields None and keeps its
        proposed geometry upstream. Nothing here decides acceptance.
        """
        self._ensure()
        import torch
        windows=plan_windows(*image.size,self.tile_side)
        refined=[None]*len(boxes);groups={}
        for index,box in enumerate(boxes):
            window=assign(box,windows)
            if window is not None:groups.setdefault(window,[]).append(index)
        for window_index,indices in groups.items():
            x1,y1,x2,y2=windows[window_index]
            view=image.crop((x1,y1,x2,y2))
            for start in range(0,len(indices),self.max_prompts):
                group=indices[start:start+self.max_prompts]
                prompts=[[[float(boxes[i][0]-x1),float(boxes[i][1]-y1),
                           float(boxes[i][2]-x1),float(boxes[i][3]-y1)] for i in group]]
                inputs=self._processor(view,input_boxes=prompts,return_tensors='pt').to(self.device)
                with torch.no_grad():
                    if self.amp and self.device!='cpu':
                        with torch.autocast('cuda',dtype=torch.float16):
                            output=self._model(**inputs,multimask_output=True)
                    else:
                        output=self._model(**inputs,multimask_output=True)
                masks=self._processor.post_process_masks(output.pred_masks.cpu(),inputs['original_sizes'].cpu())[0]
                scores=output.iou_scores.cpu()[0]
                for position,index in enumerate(group):
                    # The model scores its own masks; take the one it rates highest rather than the
                    # largest, which would bias toward over-segmentation.
                    best=int(torch.argmax(scores[position]))
                    local=mask_to_box(masks[position][best].numpy())
                    if local is None:continue
                    refined[index]={'bbox_xyxy':[local[0]+x1,local[1]+y1,local[2]+x1,local[3]+y1],
                                    'mask_quality':float(scores[position][best]),
                                    'view':f'refine-{window_index}'}
        return refined
