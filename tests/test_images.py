import pytest
from PIL import Image
from lynceus.images import normalize_image,MAX_SOURCE_PIXELS
from lynceus.geometry import transform_box,validate_box

@pytest.mark.parametrize('orientation',range(1,9))
def test_orientation(tmp_path,orientation):
    path=tmp_path/'image.jpg';exif=Image.Exif();exif[274]=orientation;Image.new('RGB',(20,10)).save(path,exif=exif)
    image,meta=normalize_image(path);assert image.size==((10,20) if orientation>=5 else (20,10))
    assert transform_box([0,0,20,10],meta['orientation_transform'],*image.size)==[0,0,*image.size]

def test_alpha(tmp_path):
    path=tmp_path/'a.png';Image.new('RGBA',(2,2),(255,0,0,0)).save(path)
    image,meta=normalize_image(path);assert image.getpixel((0,0))==(255,255,255);assert meta['alpha_conversion']=='white_composite'

def test_crop():
    assert transform_box([0,0,10,10],[[2,0,20],[0,2,30],[0,0,1]],100,100)==[20,30,40,50]
    with pytest.raises(ValueError):transform_box([0,0,10,10],[[0,0,0]]*3,100,100)


def test_multi_frame_rejected(tmp_path):
    path=tmp_path/'multi.png';frames=[Image.new('RGB',(8,8),c) for c in ('red','green','blue')]
    frames[0].save(path,save_all=True,append_images=frames[1:])
    with pytest.raises(ValueError,match='multi_frame_image'):normalize_image(path)

def test_excessive_size_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr('lynceus.images.MAX_SOURCE_PIXELS',10)
    path=tmp_path/'big.png';Image.new('RGB',(8,8)).save(path)
    with pytest.raises(ValueError,match='excessive_image_size'):normalize_image(path)
    assert MAX_SOURCE_PIXELS>0

def test_malformed_rejected(tmp_path):
    path=tmp_path/'bad.png';path.write_bytes(b'not an image')
    with pytest.raises(ValueError,match='malformed_image'):normalize_image(path)
