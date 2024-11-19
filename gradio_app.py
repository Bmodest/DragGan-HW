import os
import gradio as gr
import torch
import numpy as np
import imageio
from PIL import Image
import uuid

from draggan import utils
from draggan.draggan import drag_gan
from draggan import draggan as draggan

import face_alignment

device = 'cuda'


SIZE_TO_CLICK_SIZE = {
    1024: 8,
    512: 5,
    256: 2
}

CKPT_SIZE = {
    'stylegan2/stylegan2-ffhq-config-f.pkl': 1024,
    'stylegan2/stylegan2-cat-config-f.pkl': 256,
    'stylegan2/stylegan2-church-config-f.pkl': 256,
    'stylegan2/stylegan2-horse-config-f.pkl': 256,
    'ada/ffhq.pkl': 1024,
    'ada/afhqcat.pkl': 512,
    'ada/afhqdog.pkl': 512,
    'ada/afhqwild.pkl': 512,
    'ada/brecahad.pkl': 512,
    'ada/metfaces.pkl': 512,
    'human/stylegan_human_v2_512.pkl': 512,
    'human/stylegan_human_v2_1024.pkl': 1024,
    'self_distill/bicycles_256_pytorch.pkl': 256,
    'self_distill/dogs_1024_pytorch.pkl': 1024,
    'self_distill/elephants_512_pytorch.pkl': 512,
    'self_distill/giraffes_512_pytorch.pkl': 512,
    'self_distill/horses_256_pytorch.pkl': 256,
    'self_distill/lions_512_pytorch.pkl': 512,
    'self_distill/parrots_512_pytorch.pkl': 512,
}

DEFAULT_CKPT = 'ada/afhqcat.pkl'


def to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')


def add_points_to_image(image, points, size=5):
    image = utils.draw_handle_target_points(image, points['handle'], points['target'], size)
    return image


def on_click(image, target_point, points, size, evt: gr.SelectData):
    if target_point:
        points['target'].append([evt.index[1], evt.index[0]])
        image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
        return image, not target_point
    points['handle'].append([evt.index[1], evt.index[0]])
    image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return image, not target_point


def on_drag(model, points, max_iters, state, size, mask, lr_box):
    if len(points['handle']) == 0:
        raise gr.Error('You must select at least one handle point and target point.')
    if len(points['handle']) != len(points['target']):
        raise gr.Error('You have uncompleted handle points, try to selct a target point or undo the handle point.')
    max_iters = int(max_iters)
    W = state['W']

    handle_points = [torch.tensor(p, device=device).float() for p in points['handle']]
    target_points = [torch.tensor(p, device=device).float() for p in points['target']]

    if mask.get('mask') is not None:
        mask = Image.fromarray(mask['mask']).convert('L')
        mask = np.array(mask) == 255

        mask = torch.from_numpy(mask).float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        mask = None

    step = 0
    for image, W, handle_points in drag_gan(W, model['G'],
                                            handle_points, target_points, mask,
                                            max_iters=max_iters, lr=lr_box):
        points['handle'] = [p.cpu().numpy().astype('int') for p in handle_points]
        image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])

        state['history'].append(image)
        step += 1
        yield image, state, step



#on_smile
def on_smile(image, points, size):
    # face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = fa.get_landmarks(image)
    for pred in preds:
        right_lip= pred[48]
        right_eye_top = (pred[37] + pred[38]) / 2
        right_eye_bottom = (pred[41] + pred[40]) / 2
        right_lip_tar = (right_lip*4/5 + right_eye_bottom/5) 
        points['handle'].append([right_lip[1], right_lip[0]])   
        points['target'].append([right_lip_tar[1], right_lip_tar[0]])  

        left_lip= pred[54]
        left_eye_top = (pred[43] + pred[44]) / 2
        left_eye_bottom = (pred[47] + pred[46]) / 2
        left_lip_tar = (left_lip*4/5 + left_eye_bottom/5)
        points['handle'].append([left_lip[1], left_lip[0]])   
        points['target'].append([left_lip_tar[1], left_lip_tar[0]])   

    image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return image

#on_thin
def on_thin(image, points, size):
    # face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = fa.get_landmarks(image)
    for pred in preds:
        # 遍历每个关键点
        right_face = pred[3]
        right_lip= pred[31]
        left_face = pred[13]
        left_lip= pred[35]
        right_face_tar = (right_face + right_lip) / 2
        left_face_tar = (left_face + left_lip) / 2
        points['handle'].append([right_face[1], right_face[0]])
        points['target'].append([right_face_tar[1], right_face_tar[0]])
        points['handle'].append([left_face[1], left_face[0]])
        points['target'].append([left_face_tar[1], left_face_tar[0]])


    image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return image

#on_big
def on_big(image, points, size):
    # face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = fa.get_landmarks(image)
    for pred in preds:

        left_eye_1= pred[37]
        left_eye_2= pred[38]
        left_eye_3= pred[40]
        left_eye_4= pred[41]
        right_eye_1= pred[43]
        right_eye_2= pred[44]
        right_eye_3= pred[46]
        right_eye_4= pred[47]
        left_eye_1_tar= 2*left_eye_1 - left_eye_4
        left_eye_2_tar= 2*left_eye_2 - left_eye_3
        left_eye_3_tar= 2*left_eye_3 - left_eye_2
        left_eye_4_tar= 2*left_eye_4 - left_eye_1

        right_eye_1_tar= 2*right_eye_1 - right_eye_4
        right_eye_2_tar= 2*right_eye_2 - right_eye_3
        right_eye_3_tar= 2*right_eye_3 - right_eye_2
        right_eye_4_tar= 2*right_eye_4 - right_eye_1

        points['handle'].append([left_eye_1[1], left_eye_1[0]])
        points['target'].append([left_eye_1_tar[1], left_eye_1_tar[0]])
        points['handle'].append([left_eye_2[1], left_eye_2[0]])
        points['target'].append([left_eye_2_tar[1], left_eye_2_tar[0]])
        points['handle'].append([left_eye_3[1], left_eye_3[0]])
        points['target'].append([left_eye_3_tar[1], left_eye_3_tar[0]])
        points['handle'].append([left_eye_4[1], left_eye_4[0]])
        points['target'].append([left_eye_4_tar[1], left_eye_4_tar[0]])
        points['handle'].append([right_eye_1[1], right_eye_1[0]])
        points['target'].append([right_eye_1_tar[1], right_eye_1_tar[0]])
        points['handle'].append([right_eye_2[1], right_eye_2[0]])
        points['target'].append([right_eye_2_tar[1], right_eye_2_tar[0]])
        points['handle'].append([right_eye_3[1], right_eye_3[0]])
        points['target'].append([right_eye_3_tar[1], right_eye_3_tar[0]])
        points['handle'].append([right_eye_4[1], right_eye_4[0]])
        points['target'].append([right_eye_4_tar[1], right_eye_4_tar[0]])

        


    image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return image

#on_close
def on_close(image, points, size):
    # face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = fa.get_landmarks(image)
    for pred in preds:
        # 遍历每个关键点
        right_eye_top = (pred[37] + pred[38]) / 2
        right_eye_bottom = (pred[41] + pred[40]) / 2
        points['handle'].append([right_eye_top[1], right_eye_top[0]])   
        points['target'].append([right_eye_bottom[1], right_eye_bottom[0]])  

        left_eye_top = (pred[43] + pred[44]) / 2
        left_eye_bottom = (pred[47] + pred[46]) / 2
        points['handle'].append([left_eye_top[1], left_eye_top[0]])   
        points['target'].append([left_eye_bottom[1], left_eye_bottom[0]])   

    image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return image

def on_reset(points, image, state):
    return {'target': [], 'handle': []}, state['img'], False


def on_undo(points, image, state, size):
    image = state['img']

    if len(points['target']) < len(points['handle']):
        points['handle'] = points['handle'][:-1]
    else:
        points['handle'] = points['handle'][:-1]
        points['target'] = points['target'][:-1]

    image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return points, image, False


def on_change_model(selected, model):
    size = CKPT_SIZE[selected]

    G = draggan.load_model(utils.get_path(selected), device=device)
    model = {'G': G}
    W = draggan.generate_W(
        G,
        seed=int(1),
        device=device,
        truncation_psi=0.8,
        truncation_cutoff=8,
    )
    img, _ = draggan.generate_image(W, G, device=device)

    state = {
        'W': W,
        'img': img,
        'history': []
    }

    return model, state, img, img, size


def on_new_image(model, seed):
    G = model['G']
    W = draggan.generate_W(
        G,
        seed=int(seed),
        device=device,
        truncation_psi=0.8,
        truncation_cutoff=8,
    )
    img, _ = draggan.generate_image(W, G, device=device)

    state = {
        'W': W,
        'img': img,
        'history': []
    }

    points = {'target': [], 'handle': []}
    target_point = False
    return img, img, state, points, target_point


def on_max_iter_change(max_iters):
    return gr.update(maximum=max_iters)


def on_save_files(image, state):
    os.makedirs('draggan_tmp', exist_ok=True)
    image_name = f'draggan_tmp/image_{uuid.uuid4()}.png'
    video_name = f'draggan_tmp/video_{uuid.uuid4()}.mp4'
    imageio.imsave(image_name, image)
    imageio.mimsave(video_name, state['history'])
    return [image_name, video_name]


def on_show_save():
    return gr.update(visible=True)


def on_image_change(model, image_size, image):
    image = Image.fromarray(image)
    result = inverse_image(
        model.g_ema,
        image,
        image_size=image_size
    )
    result['history'] = []
    image = to_image(result['sample'])
    points = {'target': [], 'handle': []}
    target_point = False
    return image, image, result, points, target_point


def on_mask_change(mask):
    return mask['image']


def on_select_mask_tab(state):
    img = to_image(state['sample'])
    return img


def main():
    torch.cuda.manual_seed(25)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # DragGAN
            
            Unofficial implementation of [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
            
            [Our Implementation](https://github.com/Zeqiang-Lai/DragGAN) | [Official Implementation](https://github.com/XingangPan/DragGAN) (Not released yet)

            ## Tutorial
            
            1. (Optional) Draw a mask indicate the movable region.
            2. Setup a least one pair of handle point and target point.
            3. Click "Drag it". 
            
            ## Hints
            
            - Handle points (Blue): the point you want to drag.
            - Target points (Red): the destination you want to drag towards to.
            
            ## Primary Support of Custom Image.
            
            - We now support dragging user uploaded image by GAN inversion.
            - **Please upload your image at `Setup Handle Points` pannel.** Upload it from `Draw a Mask` would cause errors for now.
            - Due to the limitation of GAN inversion, 
                - You might wait roughly 1 minute to see the GAN version of the uploaded image.
                - The shown image might be slightly difference from the uploaded one.
                - It could also fail to invert the uploaded image and generate very poor results.
                - Idealy, you should choose the closest model of the uploaded image. For example, choose `stylegan2-ffhq-config-f.pkl` for human face. `stylegan2-cat-config-f.pkl` for cat.
                
            > Please fire an issue if you have encounted any problem. Also don't forgot to give a star to the [Official Repo](https://github.com/XingangPan/DragGAN), [our project](https://github.com/Zeqiang-Lai/DragGAN) could not exist without it.
            """,
        )
        G = draggan.load_model(utils.get_path(DEFAULT_CKPT), device=device)
        model = gr.State({'G': G})
        W = draggan.generate_W(
            G,
            seed=int(1),
            device=device,
            truncation_psi=0.8,
            truncation_cutoff=8,
        )
        img, F0 = draggan.generate_image(W, G, device=device)

        state = gr.State({
            'W': W,
            'img': img,
            'history': []
        })
        points = gr.State({'target': [], 'handle': []})
        size = gr.State(CKPT_SIZE[DEFAULT_CKPT])
        target_point = gr.State(False)

        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Accordion("Model"):
                    model_dropdown = gr.Dropdown(choices=list(CKPT_SIZE.keys()), value=DEFAULT_CKPT,
                                                 label='StyleGAN2 model')
                    seed = gr.Number(value=1, label='Seed', precision=0)
                    new_btn = gr.Button('New Image')
                with gr.Accordion('Drag'):
                    with gr.Row():
                        lr_box = gr.Number(value=2e-3, label='Learning Rate')
                        max_iters = gr.Slider(1, 500, 20, step=1, label='Max Iterations')

                    with gr.Row():
                        with gr.Column(min_width=100):
                            reset_btn = gr.Button('Reset All')
                        with gr.Column(min_width=100):
                            undo_btn = gr.Button('Undo Last')
                    with gr.Row():
                        btn = gr.Button('Drag it', variant='primary')
                with gr.Accordion("一键修改"):
                    btn_smile = gr.Button('微笑')
                    btn_thin = gr.Button('瘦脸')
                    btn_big = gr.Button('大眼')
                    btn_close = gr.Button('闭眼')
                
                with gr.Accordion('Save', visible=False) as save_panel:
                    files = gr.Files(value=[])

                progress = gr.Slider(value=0, maximum=20, label='Progress', interactive=False)

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab('Setup Handle Points', id='input'):
                        image = gr.Image(img).style(height=512, width=512)
                    with gr.Tab('Draw a Mask', id='mask') as masktab:
                        mask = gr.ImageMask(img, label='Mask').style(height=512, width=512)

        image.select(on_click, [image, target_point, points, size], [image, target_point])
        image.upload(on_image_change, [model, size, image], [image, mask, state, points, target_point])
        mask.upload(on_mask_change, [mask], [image])
        btn.click(on_drag, inputs=[model, points, max_iters, state, size, mask, lr_box], outputs=[image, state, progress]).then(
            on_show_save, outputs=save_panel).then(
            on_save_files, inputs=[image, state], outputs=[files]
        )
        btn_smile.click(on_smile, inputs=[image, points, size], outputs=[image]).then(
            on_drag, inputs=[model, points, max_iters, state, size, mask, lr_box], outputs=[image, state, progress]).then(
            on_show_save, outputs=save_panel).then(
            on_save_files, inputs=[image, state], outputs=[files]
        )
        btn_thin.click(on_thin, inputs=[image, points, size], outputs=[image]).then(
            on_drag, inputs=[model, points, max_iters, state, size, mask, lr_box], outputs=[image, state, progress]).then(
            on_show_save, outputs=save_panel).then(
            on_save_files, inputs=[image, state], outputs=[files]
        )
        btn_big.click(on_big, inputs=[image, points, size], outputs=[image]).then(
            on_drag, inputs=[model, points, max_iters, state, size, mask, lr_box], outputs=[image, state, progress]).then(
            on_show_save, outputs=save_panel).then(
            on_save_files, inputs=[image, state], outputs=[files]
        )
        btn_close.click(on_close, inputs=[image, points, size], outputs=[image]).then(
            on_drag, inputs=[model, points, max_iters, state, size, mask, lr_box], outputs=[image, state, progress]).then(
            on_show_save, outputs=save_panel).then(
            on_save_files, inputs=[image, state], outputs=[files]
        )
        reset_btn.click(on_reset, inputs=[points, image, state], outputs=[points, image, target_point])
        undo_btn.click(on_undo, inputs=[points, image, state, size], outputs=[points, image, target_point])
        model_dropdown.change(on_change_model, inputs=[model_dropdown, model], outputs=[model, state, image, mask, size])
        new_btn.click(on_new_image, inputs=[model, seed], outputs=[image, mask, state, points, target_point])
        max_iters.change(on_max_iter_change, inputs=max_iters, outputs=progress)
        masktab.select(lambda: gr.update(value=None), outputs=[mask]).then(on_select_mask_tab, inputs=[state], outputs=[mask])
    return demo


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('-p', '--port', type=int, default=None)
    parser.add_argument('--ip', default=None)
    args = parser.parse_args()
    device = args.device
    demo = main()
    print('Successfully loaded, starting gradio demo')
    demo.queue(concurrency_count=1, max_size=20).launch(share=args.share, server_name=args.ip, server_port=args.port)
