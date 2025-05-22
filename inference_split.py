import os
import pickle
import torch
import copy
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from deepspeed import init_inference

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

model_path = "deepseek-ai/deepseek-vl2-small"
size_per_device_GB = 4
num_device = torch.cuda.device_count()
max_new_tokens = 512

def split_model(model_name):
    device_map = {}
    model_splits = {
        'deepseek-ai/deepseek-vl2-tiny': [12], # 1 GPU
        'deepseek-ai/deepseek-vl2-small': [13, 14], # 2 GPU
        'deepseek-ai/deepseek-vl2': [10, 10, 10], # 3 GPU for 27b
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers =  sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map

def get_model():
    # specify the path to the model
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    device_map = split_model(model_path)
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    ).eval()
    return vl_gpt, vl_chat_processor, tokenizer

def process_datas_from_pkl(path_GenAD, path_pkl, path_pkl_new, vl_gpt, vl_chat_processor, tokenizer):
    # 读取pkl的数据
    with open(os.path.join(path_pkl), 'rb') as f:
        datas_from_pkl = pickle.load(f)

    # 相机位置和ID
    camera_positions = {
        'CAM_FRONT': 0,
        'CAM_FRONT_RIGHT': 1,
        'CAM_BACK_RIGHT': 2,
        'CAM_BACK': 3,
        'CAM_BACK_LEFT': 4,
        'CAM_FRONT_LEFT': 5
    }

    # 遍历当前的所有info
    infos = []
    for datas_info in tqdm(datas_from_pkl['infos']):
        # 获取当前的图像路径
        cams = datas_info['cams']
        images_path = []
        for cp in camera_positions:
            # 获取当前相机位置对应的图片路径
            data_path = cams[cp]['data_path']
            # 将路径中的GenAD字段去除
            data_path_ = os.path.join(path_GenAD, data_path)
            assert os.path.exists(data_path_)
            images_path.append(data_path_)
        
        # 设定输入的文本
        system_prompt = (
            "You are an intelligent autonomous driving assistant. "
            "Your goal is to generate a comprehensive and unified textual description of the driving scene "
            "based on multiple images from different surrounding cameras on the ego vehicle. "
            "You must reason about the 3D spatial layout and include all critical elements in the environment."
        )
        # 预设部分content
        content_preset = (
            "Here are six images captured simultaneously from a 360-degree surround-view camera system on the ego vehicle: " + "<image>" * 6 + ". "
            "These correspond to: CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_FRONT_LEFT. "
            "Generate a single unified paragraph that summarizes the current scene from all viewpoints. "
            "Do not describe images individually. Instead, integrate the visual information to reason about the environment, positions of objects, and scene layout relative to the ego vehicle."
        )
        # 设置不同key
        key_all = (
            "Describe the entire driving scene in detail, covering all visible elements. "
            "Include dynamic objects, their types, positions, and motions, as well as static elements like road structures, "
            "traffic signs, lane markings, intersections, and any other relevant environmental features. "
            "Additionally, assess and describe the current traffic congestion or road occupancy around the ego vehicle. "
            "Also, describe the expected or planned motion state of the ego vehicle in the near future. "
            "Provide a holistic understanding of the spatial layout, interactions, and ego vehicle's upcoming movements."
        )
        key_motion = (
            "Focus exclusively on other dynamic objects in the scene. "
            "Describe their types (e.g., vehicles, pedestrians, cyclists), relative positions to the ego vehicle, "
            "and provide predictions about their possible motions or behaviors. "
        )
        key_map = (
            "Describe only the static road and ground features. "
            "Include lane markings, road boundaries, intersections, pedestrian crossings, traffic signs, and any other relevant road surface details. "
            "Ignore all other dynamic objects."
        )

        keys = [key_all, key_motion, key_map]


        # 针对不同对象进行描述
        contents = []
        answers = []
        answers_token = []
        for key in keys:
            # 获取交互文本
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f'{key}\n\n{content_preset}',
                    "images": images_path,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=system_prompt
            ).to(vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False
            )

            answer_token = outputs[0]
            assert not (answer_token == 0.0).any(), 'There are [0] in answer_token'
            assert len(answer_token) <= max_new_tokens, "The length of the answer is too long."
            answer_token = answer_token.cpu().tolist()
            answer = tokenizer.decode(answer_token, skip_special_tokens=True)
            answer_token = answer_token + (max_new_tokens - len(answer_token)) * [None]  # 补充到统一的长度

            contents.append(prepare_inputs['sft_format'])
            answers.append(answer)
            answers_token.append(answer_token)
        # 保存生成的文本
        datas_info['description'] = {
            'contents': contents,
            'answers': answers,
            'answers_token': answers_token
        }
        infos.append(datas_info)

    # 保存成pkl文件
    with open(os.path.join(path_pkl_new), 'wb') as f:
        datas_from_pkl['infos'] = infos
        pickle.dump(datas_from_pkl, f)

if __name__ == '__main__':
    # load the model
    models = get_model()

    path_GenAD = '../GenAD'
    # path_folder = '../GenAD/data/nuscenes/'
    path_folder = '../GenAD/data/infos/full/'
    datas_pkls = [
        # 'vad_nuscenes_infos_temporal_train.pkl', 
        'vad_nuscenes_infos_temporal_val.pkl',
        'vad_nuscenes_infos_temporal_test.pkl'
    ]
    datas_pkls_new = [
        # 'vad_nuscenes_infos_temporal_train_with_description.pkl', 
        'vad_nuscenes_infos_temporal_val_with_description.pkl',
        'vad_nuscenes_infos_temporal_test_with_description.pkl'
    ]

    for data_pkls, datas_pkl_new in zip(datas_pkls, datas_pkls_new):
        with torch.no_grad():
            process_datas_from_pkl(
                path_GenAD, 
                os.path.join(path_folder, data_pkls),
                os.path.join(path_folder, datas_pkl_new),
                *models
            )
