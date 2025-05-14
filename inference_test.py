import os
import pickle
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

def get_model():
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, vl_chat_processor, tokenizer

def process_datas_from_pkl(path_GenAD, path_pkl, path_pkl_new, vl_gpt, vl_chat_processor, tokenizer):
    # 读取pkl的数据
    with open(os.path.join(path_GenAD, path_pkl), 'rb') as f:
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
        key_all = 'The description should include information such as future predictions, objects, traffic signs, traffic signals and the road structure in the scene. '
        key_motion = 'The description should only focuse on the motion information of other objects and ignore the map information.' \
                     'Please detect and describe the types and possible motion states of the objects in the scene.'
        key_map = 'The description should only focuse on the map information, which includes [lane divider, road boundary and pedestrian crossing], and ignore the object information.' \
                  'Please detect and describe the structure, types, traffic signs, traffic rules and existing signs of the current road.'
        keys = [key_all, key_motion, key_map]
        preset = ""
        for cp in camera_positions:
            preset += f"This is image from {cp}: <image>\n"
        preset = preset + \
                "These pictures are images from six perspectives provided by the ego vehicle. " \
                "You need to parse images from different perspectives and generate a detailed textual description of the current scene. " \
                "Please pay attention to distinguishing images from different perspectives " \
                "and reasoning out the correct positional relationship between the objects and the ego vehicle in the picture. " \

        # 针对不同对象进行描述
        contents = []
        answers = []
        for key in keys:
            content = preset + key
            # 获取交互文本
            conversation = [
                {
                    "role": "<|User|>",
                    "content": content,
                    "images": images_path,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
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
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

            contents.append(content)
            answers.append(answer)
        # 保存生成的文本
        datas_info['description'] = {
            'contents': contents,
            'answers': answers
        }
        infos.append(datas_info)

    # 保存成pkl文件
    with open(os.path.join(path_GenAD, path_pkl_new), 'wb') as f:
        datas_from_pkl['infos'] = infos
        pickle.dump(datas_from_pkl, f)

if __name__ == '__main__':
    # load the model
    vl_gpt, vl_chat_processor, tokenizer = get_model()

    path_GenAD = '../GenAD'
    path_datas_pkl = [
        'data/nuscenes/vad_nuscenes_infos_temporal_train.pkl', 
        'data/nuscenes/vad_nuscenes_infos_temporal_val.pkl'
        # 'data/nuscenes/vad_nuscenes_infos_temporal_test.pkl'
    ]
    path_datas_pkl_new = [
        'data/nuscenes/vad_nuscenes_infos_temporal_train_with_description.pkl', 
        'data/nuscenes/vad_nuscenes_infos_temporal_val_with_description.pkl',
        # 'data/nuscenes/vad_nuscenes_infos_temporal_test_with_description.pkl'
    ]

    for path_pkl, path_pkl_new in zip(path_datas_pkl, path_datas_pkl_new):
            process_datas_from_pkl(path_GenAD, path_pkl, path_pkl_new, vl_gpt, vl_chat_processor, tokenizer)
