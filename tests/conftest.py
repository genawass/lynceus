import json
import socket
import pytest
from PIL import Image
from lynceus.artifacts import digest

LOOPBACK=('127.0.0.1','::1','localhost')


def _is_loopback(address):
    return isinstance(address,tuple) and address and address[0] in LOOPBACK


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """No test may reach the network, because no part of this system may.

    Loopback is the one exception, and it is not a weakening: the review server is a local tool for
    building a reference panel, outside the annotator's inference path, and testing it means
    talking to a socket on this host. Anything with a destination off the host still fails, so the
    guarantee the suite exists to protect is untouched.
    """
    real_connect,real_create=socket.socket.connect,socket.create_connection
    def connect(self,address,*a,**k):
        if not _is_loopback(address):raise AssertionError('network forbidden')
        return real_connect(self,address,*a,**k)
    def create_connection(address,*a,**k):
        if not _is_loopback(address):raise AssertionError('network forbidden')
        return real_create(address,*a,**k)
    monkeypatch.setattr(socket.socket,'connect',connect)
    monkeypatch.setattr(socket,'create_connection',create_connection)

@pytest.fixture
def staged(tmp_path):
    root=tmp_path/'bundle';model=root/'model';model.mkdir(parents=True)
    assets=[]
    for name in ['config.json','preprocessor_config.json','tokenizer_config.json','model.safetensors']:
        path=model/name;path.write_text('{}');assets.append({'path':'model/'+name,'sha256':digest(path.read_bytes())})
    bundle=root/'bundle.json';bundle.write_text(json.dumps({'schema_version':'lynceus.bundle/1.0','adapter':'owlv2','qualification':'unqualified','model_dir':'model','dependencies':{},'assets':assets}))
    image=tmp_path/'image.png';Image.new('RGB',(100,60),'white').save(image)
    return image,bundle,tmp_path/'run'

class FakeAdapter:
    def predict(self,image):return [{'bbox_xyxy':[1,2,20,30],'label':'cup','model_score':.99}]

@pytest.fixture
def completed(staged):
    from lynceus.pipeline import annotate
    image,bundle,run=staged
    return annotate(image,bundle,run,adapter=FakeAdapter()),run


@pytest.fixture
def network_is_still_denied():
    """Handed to a test that wants to assert the deny fixture is doing its job."""
    return ('example.com', 80)
