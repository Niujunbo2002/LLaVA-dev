
def prepare_image_features_for_truncated(new_input_embeds,input_ids,max_length,IMAGE_TOKEN_INDEX,image_features):
    B=len(new_input_embeds)
    # 记录每个图像标记的全局索引和位置
    image_positions = []
    for batch_idx in range(B):
        positions = (input_ids[batch_idx] == IMAGE_TOKEN_INDEX).nonzero().squeeze().tolist()
        if isinstance(positions, int):  # 处理单元素情况
            positions = [positions]
        image_positions.append(positions)

    # 生成全局索引映射 (假设特征按标记出现顺序排列)
    global_indices = []
    current_idx = 0
    for positions in image_positions:
        global_indices.append(list(range(current_idx, current_idx + len(positions))))
        current_idx += len(positions)

    # 收集需要保留的特征索引
    selected_indices = []
    for batch_idx in range(B):
        valid_positions = [pos for pos in image_positions[batch_idx] if pos < max_length]
        selected_indices.extend([global_indices[batch_idx][i] for i, pos in enumerate(image_positions[batch_idx]) if pos < max_length])
    # print(selected_indices)
    return image_features[selected_indices]