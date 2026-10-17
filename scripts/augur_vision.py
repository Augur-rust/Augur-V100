"""Bounded PNG/JPEG data URLs and canonical RGB image identity; no remote fetch."""
from __future__ import annotations
import base64
import copy
import hashlib
import io
import math
import struct
from PIL import Image, ImageOps, UnidentifiedImageError
import qwen35_api as A
import qwen35_chat as C

MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_PIXELS = 16_777_216
MAX_IMAGE_PIXELS = 1_048_576
# Resource budget across the transcript, independent of image count. Keep each
# image's resize stable as history grows; never drop/re-resize old screenshots.
MAX_TOTAL_IMAGE_PIXELS = 32 * MAX_IMAGE_PIXELS
MAX_IMAGE_TOKENS = MAX_TOTAL_IMAGE_PIXELS // 1024
MAX_REQUEST_BYTES = 256 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS

def resized_extent(height, width, detail='auto'):
    """32-aligned smart resize, following Qwen/NInfer min/max-area semantics.

    Auto and high request at least512 merged tokens for screenshot/OCR inputs;
    low preserves the explicit64-token option. Each image is budgeted separately
    so retained-image identity never depends on later additions to the transcript.
    These budgets are product policy, not checkpoint model geometry.
    """
    C.require(detail in ('auto', 'low', 'high'), 'image detail must be auto, low or high')
    C.require(height > 0 and width > 0 and max(height, width) / min(height, width) <= 200,
              'image aspect ratio bound')
    minimum = {'low': 65536, 'auto': 524288, 'high': 524288}[detail]
    h, w = round(height / 32) * 32, round(width / 32) * 32
    if h * w > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / (height * width))
        h, w = max(32, math.floor(height * scale / 32) * 32), max(32, math.floor(width * scale / 32) * 32)
    elif h * w < minimum:
        scale = math.sqrt(minimum / (height * width))
        h, w = math.ceil(height * scale / 32) * 32, math.ceil(width * scale / 32) * 32
    C.require(0 < h * w <= MAX_IMAGE_PIXELS, 'aligned image pixel budget')
    return h, w

def decode(url, detail='auto'):
    C.require(isinstance(url, str) and len(url) <= MAX_SOURCE_BYTES * 4 // 3 + 128, 'image data URL bound')
    prefix, sep, payload = url.partition(',')
    C.require(sep and prefix in ('data:image/png;base64', 'data:image/jpeg;base64'), 'PNG/JPEG base64 data URL required')
    try:
        raw = base64.b64decode(payload, validate=True)
        C.require(0 < len(raw) <= MAX_SOURCE_BYTES, 'image encoded byte bound')
        with Image.open(io.BytesIO(raw)) as source:
            C.require(source.format == ('PNG' if 'png' in prefix else 'JPEG'), 'image MIME/content mismatch')
            C.require(getattr(source, 'n_frames', 1) == 1 and source.width > 0 and source.height > 0
                      and source.width * source.height <= MAX_SOURCE_PIXELS, 'image source geometry/frames')
            image = ImageOps.exif_transpose(source).convert('RGB')
        height, width = resized_extent(image.height, image.width, detail)
        if image.size != (width, height): image = image.resize((width, height), Image.Resampling.BICUBIC)
        rgb = image.tobytes()
        return dict(width=width, height=height, rgb8_hex=rgb.hex(), rgb8_sha256=hashlib.sha256(rgb).hexdigest())
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise C.ProtocolError('invalid image: ' + str(error)) from error

def collect(messages):
    return [part['image'] for message in messages if isinstance(message['content'], list)
            for part in message['content'] if part.get('type') == 'image']

def identity(images):
    h = hashlib.sha256(b'QUASAR-IMAGE-INPUT-RGB8-V1\0')
    for image in images:
        h.update(struct.pack('<QQ', image['height'], image['width']))
        h.update(bytes.fromhex(image['rgb8_sha256']))
    return h.hexdigest()

def render(template, messages, tools=None, generation=True, enable_thinking=False, reasoning_effort=None):
    text = C.ChatTemplate.render(template, messages, tools=tools, add_generation_prompt=generation,
                                 enable_thinking=enable_thinking, reasoning_effort=reasoning_effort)
    images = collect(messages)
    C.require(text.count('<|image_pad|>') == len(images), 'image template placeholder binding')
    parts = text.split('<|image_pad|>')
    return ''.join(parts[i] + '<|image_pad|>' * (image['height'] * image['width'] // 1024)
                   for i, image in enumerate(images)) + parts[-1]

def validate_request(payload, model):
    messages = payload.get('messages') if isinstance(payload, dict) else None
    has_images = isinstance(messages, list) and any(isinstance(m, dict) and isinstance(m.get('content'), list)
        and any(isinstance(p, dict) and p.get('type') == 'image_url' for p in m['content']) for m in messages)
    if not has_images: return A.validate_request(payload, model)
    # The existing tool validator still authenticates all roles/call IDs and
    # schemas. Replace only user image content during that validation, then
    # restore bounded decoded pixels; never bypass tool-history validation.
    safe = copy.deepcopy(payload); normalized = []; total_pixels = 0
    try:
        for i, message in enumerate(messages):
            A.need(isinstance(message, dict), 'Invalid message.', 'messages')
            content = message.get('content')
            if not isinstance(content, list): normalized.append(copy.deepcopy(message)); continue
            parts = []
            for part in content:
                A.need(isinstance(part, dict), 'Invalid content part.', 'messages')
                if part.get('type') == 'image_url':
                    A.need(message.get('role') == 'user' and set(part) == {'type', 'image_url'}
                           and isinstance(part['image_url'], dict) and set(part['image_url']) <= {'url', 'detail'}
                           and part['image_url'].get('detail', 'auto') in ('auto', 'low', 'high'), 'Invalid image part.', 'messages')
                    image=decode(part['image_url'].get('url'),part['image_url'].get('detail','auto'))
                    total_pixels += image['width'] * image['height']
                    A.need(total_pixels <= MAX_TOTAL_IMAGE_PIXELS,
                           f'Image history exceeds the total resized-pixel budget ({total_pixels} > {MAX_TOTAL_IMAGE_PIXELS}); no images were removed. Use lower-resolution source images or start a new conversation.', 'messages')
                    parts.append(dict(type='image', image=image))
                else:
                    A.need(set(part) == {'type', 'text'} and part.get('type') == 'text'
                           and isinstance(part.get('text'), str), 'Invalid text part.', 'messages')
                    # DSH coalesces adjacent image/user messages with whitespace
                    # text parts. Preserve separators; validate nonempty text and
                    # reject a wholly empty message below, not each fragment.
                    if part['text'].strip(): C.ChatTemplate.message('user', part['text'])
                    parts.append(copy.deepcopy(part))
            A.need(any(p.get('type') == 'image' or p.get('text', '').strip() for p in parts),
                   'Empty content parts.', 'messages')
            normalized.append(dict(message, content=parts))
            safe['messages'][i] = dict(message, content='Image input')
        validated = A.validate_request(safe, model)
        images = collect(normalized)
        A.need(images, 'At least one image is required.', 'messages')
        for message, normalized_message in zip(validated['messages'], normalized):
            if isinstance(normalized_message['content'], list): message['content'] = normalized_message['content']
        return validated
    except (C.ProtocolError, ValueError, TypeError, KeyError) as error:
        raise A.ApiError(str(error), param='messages') from error
