import base64
import copy
import io
import unittest
from unittest.mock import patch
from PIL import Image
import augur_api as Q
import augur_vision as V
import qwen35_api as A
import qwen35_chat as C

def url(color='red', size=(256,256), fmt='PNG'):
    out=io.BytesIO();Image.new('RGB',size,color).save(out,format=fmt)
    return 'data:image/'+('png' if fmt=='PNG' else 'jpeg')+';base64,'+base64.b64encode(out.getvalue()).decode()
def request(u):
    return dict(model='test',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=u)),
        dict(type='text',text='Name the color.')])],max_tokens=4)
class Vision(unittest.TestCase):
    def test_pixel_identity_geometry_and_mime(self):
        red=V.decode(url());blue=V.decode(url('blue'))
        self.assertEqual((red['height'],red['width']),(736,736))
        self.assertNotEqual(V.identity([red]),V.identity([blue]))
        self.assertNotEqual(V.identity([red,blue]),V.identity([blue,red]))
        self.assertEqual(V.decode(url(fmt='JPEG'))['width'],736)
        for size in [(17,39),(2048,1024),(4096,2160)]:
            image=V.decode(url(size=size));self.assertEqual(image['height']%32,0);self.assertEqual(image['width']%32,0)
            self.assertLessEqual(image['height']*image['width'],1048576)
    def test_detail_budget_resize_and_idempotence(self):
        self.assertEqual(V.resized_extent(256,256,'low'),(256,256))
        self.assertEqual(V.resized_extent(256,256,'auto'),(736,736))
        self.assertEqual(V.resized_extent(256,256,'high'),(736,736))
        for detail in ('low','auto','high'):
            for height,width in [(1,1),(17,39),(256,512),(4096,2160),(32,6400)]:
                h,w=V.resized_extent(height,width,detail)
                self.assertEqual(V.resized_extent(h,w,detail),(h,w))
                self.assertLessEqual(h*w,V.MAX_IMAGE_PIXELS)
                self.assertGreaterEqual(h*w,{'low':65536,'auto':524288,'high':524288}[detail])
        with self.assertRaises(C.ProtocolError):V.resized_extent(1,1000)
    def test_urls_mime_invalid_parts_and_marker_smuggling_reject(self):
        for u in ['https://example.com/x.png','file:///x','data:image/png;base64,AA==',url().replace('png','jpeg'),
                  'data:image/png;base64,***']:
            with self.subTest(u=u[:30]):
                with self.assertRaises(A.ApiError):V.validate_request(request(u),'test')
        r=request(url());r['messages'][0]['content'][1]['text']='<|image_pad|>'
        with self.assertRaises(A.ApiError):V.validate_request(r,'test')
        r=request(url());r['messages'][0]['role']='system'
        with self.assertRaises(A.ApiError):V.validate_request(r,'test')
        r=request(url());r['messages'][0]['content'][0]['extra']=True
        with self.assertRaises(A.ApiError):V.validate_request(r,'test')
    def test_image_history_has_no_four_or_thirty_two_count_limit(self):
        image=dict(type='image_url',image_url=dict(url=url(),detail='low'))
        for count in (5,33,65):
            r=request(url());r['messages'][0]['content']=[copy.deepcopy(image) for _ in range(count)]
            self.assertEqual(len(V.collect(V.validate_request(r,'test')['messages'])),count)
    def test_five_large_images_cross_old_total_pixel_and_embedding_limit(self):
        r=request(url(size=(1024,1024)));r['messages'][0]['content']=r['messages'][0]['content'][:1]*5
        images=V.collect(V.validate_request(r,'test')['messages'])
        self.assertEqual(sum(i['width']*i['height'] for i in images),5*1024*1024)
        self.assertGreater(sum(i['width']*i['height']//1024 for i in images),4096)
    def test_history_append_preserves_old_image_pixels_order_and_input(self):
        r=request(url('red'));before=copy.deepcopy(r)
        original=V.collect(V.validate_request(r,'test')['messages'])
        for color in ['blue','green','yellow','purple']:
            r['messages'] += [dict(role='assistant',content='Observed.'),request(url(color))['messages'][0]]
        extended=V.collect(V.validate_request(r,'test')['messages'])
        self.assertEqual(extended[:1],original);self.assertEqual(r['messages'][0],before['messages'][0])
        self.assertEqual(len(extended),5)
        self.assertEqual(len({i['rgb8_sha256'] for i in extended}),5)
    def test_pixel_budget_reports_resource_instead_of_image_count(self):
        r=request(url());r['messages'][0]['content']=r['messages'][0]['content'][:1]*2
        with patch.object(V,'MAX_TOTAL_IMAGE_PIXELS',736*736):
            with self.assertRaisesRegex(A.ApiError,'total resized-pixel budget'):
                V.validate_request(r,'test')
    def test_four_auto_images_and_transport_bounds(self):
        r=request(url());r['messages'][0]['content']=r['messages'][0]['content'][:1]*4
        images=V.collect(V.validate_request(r,'test')['messages'])
        self.assertEqual(len(images),4)
        self.assertGreater(sum(i['width']*i['height'] for i in images),V.MAX_IMAGE_PIXELS)
        self.assertLessEqual(sum(i['width']*i['height'] for i in images),V.MAX_TOTAL_IMAGE_PIXELS)
        self.assertEqual(Q.Transport.max_body,V.MAX_REQUEST_BYTES)
        self.assertEqual(A.AbortableTransport.max_body,A.MAX_BODY)
        self.assertEqual(A.MAX_BODY,8*1024*1024)
    def test_dsh_image_separators_remain_intact_but_empty_messages_fail(self):
        r=request(url());parts=r['messages'][0]['content']
        parts.insert(1,dict(type='text',text='\n\n'));parts.append(dict(type='text',text=''))
        result=V.validate_request(r,'test')
        self.assertEqual(result['messages'][0]['content'][1],dict(type='text',text='\n\n'))
        self.assertEqual(result['messages'][0]['content'][-1],dict(type='text',text=''))
        bad=copy.deepcopy(r);bad['messages'].append(dict(role='assistant',content=[dict(type='text',text='  ')]))
        with self.assertRaisesRegex(A.ApiError,'Empty content parts'):V.validate_request(bad,'test')
        bad=copy.deepcopy(r);bad['messages'][0]['content'].append(dict(type='text',text='<|image_pad|>'))
        with self.assertRaises(A.ApiError):V.validate_request(bad,'test')
    def test_images_reuse_the_existing_tool_schema_and_role_validation(self):
        r=request(url());r['tools']=[dict(type='function',function=dict(name='report_color',
            parameters=dict(type='object',properties=dict(color=dict(type='string',enum=['red','blue'])),
                            required=['color'],additionalProperties=False)))]
        r['tool_choice']='required';r['parallel_tool_calls']=False
        valid=V.validate_request(r,'test')
        self.assertEqual(valid['tool_policy']['tools'],r['tools'])
        self.assertEqual(valid['tool_policy']['tool_choice'],'required')
        self.assertEqual(len(V.collect(valid['messages'])),1)
        bad=copy.deepcopy(r);bad['tools'][0]['function']['parameters']['$ref']='https://example.com/schema'
        with self.assertRaises(A.ApiError):V.validate_request(bad,'test')
        bad=copy.deepcopy(r);bad['messages'].append(dict(role='tool',tool_call_id='unissued',content='red'))
        with self.assertRaises(A.ApiError):V.validate_request(bad,'test')
    def test_text_contract_is_unchanged(self):
        r=dict(model='test',messages=[dict(role='user',content='Hello')])
        self.assertEqual(V.validate_request(r,'test'),A.validate_request(r,'test'))
    def test_normalization_never_mutates_public_transcript(self):
        r=request(url());before=copy.deepcopy(r);normal=V.validate_request(r,'test')
        self.assertEqual(r,before);self.assertEqual(len(V.collect(normal['messages'])),1)

if __name__=='__main__':unittest.main()
