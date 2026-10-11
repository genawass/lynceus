"""Crop embeddings from a local CLIP checkpoint, and nearest-class retrieval over a reference bank.

This is not a detector and not a verifier. It answers one question about a box that already exists:
of the classes a bank holds examples of, which does this crop look most like. That makes it a
candidate for the step conservative naming cannot take -- resolving a parent fallback into a child
class -- without asking a second detector whether anything is there.

Two properties make it worth measuring separately from the verifier. It is retrieval rather than
generation, so the classes it can emit are exactly the classes the bank contains and it cannot
invent one. And the bank is built from reference crops of the training split, so what it compares
against is human-labelled examples of this domain rather than a text prompt whose phrasing is its
own uncontrolled variable.

The bank is the experiment's weak point and is kept explicit: crops come from the split declared
under A3, the sequences shared with the evaluation panel are excluded, and the bank's contents
travel with any measurement made through it.
"""
import os

REQUIRED_DEPENDENCIES=('torch','transformers')


class ClipEmbedder:
    """Local CLIP image tower over crops, with cosine retrieval against a per-class bank."""

    def __init__(self,model_dir,device=None,batch=32,context=1.5):
        self.model_dir=str(model_dir)
        self.batch=batch
        # A tight crop of a twelve-pixel car is twelve pixels; context is what makes it legible,
        # and how much context is a declared parameter rather than a default worth hiding.
        self.context=context
        self._requested_device=device;self._model=None

    def _ensure(self):
        if self._model is not None:return
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection
        self.device=self._requested_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._processor=CLIPImageProcessor.from_pretrained(self.model_dir,local_files_only=True)
        self._model=(CLIPVisionModelWithProjection.from_pretrained(self.model_dir,local_files_only=True)
                     .eval().to(self.device))

    def capabilities(self):
        self._ensure()
        return {'device':self.device,'checkpoint':self.model_dir,'batch':self.batch,
                'context':self.context,'role':'crop_embedding_and_retrieval',
                'emits':'a class the bank contains, never a new class'}

    def crop(self,image,box):
        """The box with declared context around it, clipped to the image."""
        x1,y1,x2,y2=box
        width,height=(x2-x1),(y2-y1)
        pad_x,pad_y=width*(self.context-1)/2,height*(self.context-1)/2
        return image.crop((max(0,int(x1-pad_x)),max(0,int(y1-pad_y)),
                           min(image.width,int(x2+pad_x)),min(image.height,int(y2+pad_y))))

    def embed(self,image,boxes):
        """L2-normalized embeddings, one row per box, in the order given."""
        self._ensure()
        import torch
        rows=[]
        for start in range(0,len(boxes),self.batch):
            crops=[self.crop(image,box) for box in boxes[start:start+self.batch]]
            inputs=self._processor(images=crops,return_tensors='pt').to(self.device)
            with torch.inference_mode():
                embeds=self._model(**inputs).image_embeds
            rows.append(torch.nn.functional.normalize(embeds,dim=-1).float().cpu())
        return torch.cat(rows) if rows else torch.zeros(0,1)

    def nearest(self,embeddings,bank,bank_labels,k=5):
        """Per row: the class of the k nearest bank crops by vote, with the winning similarity.

        A vote over k neighbours rather than a single nearest crop, because one bank crop can be
        mislabelled or atypical and a vote makes that cost one vote. A nearest-centroid rule was
        the alternative and is worse on a class whose appearance is multimodal, which `vehicle`
        covering cars, vans, trucks and buses certainly is.
        """
        import torch
        if not len(embeddings) or not len(bank):return []
        similarity=embeddings@bank.T
        top=torch.topk(similarity,min(k,len(bank)),dim=-1)
        out=[]
        for values,indices in zip(top.values.tolist(),top.indices.tolist()):
            votes={}
            for value,index in zip(values,indices):
                label=bank_labels[index]
                votes.setdefault(label,[0,0.0])
                votes[label][0]+=1;votes[label][1]=max(votes[label][1],value)
            label,(count,best)=max(votes.items(),key=lambda item:(item[1][0],item[1][1]))
            out.append({'label':label,'votes':count,'k':min(k,len(bank)),
                        'similarity':round(best,6),
                        'runner_up':sorted(((v[0],v[1],name) for name,v in votes.items()),
                                           reverse=True)[1][2] if len(votes)>1 else None})
        return out
