!pip install genesis-world torch

import genesis as gs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# シミュレーション環境のセットアップ
# ==========================================
import genesis as gs
if not gs._initialized:
    gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

# 物体の位置
pos_A = np.array([0.5, 0.2, 0.025])
pos_B = np.array([0.5, -0.2, 0.025])
peg_entity = scene.add_entity(gs.morphs.Cylinder(pos=pos_A, radius=0.015, height=0.05), material=gs.materials.Rigid())

cam = scene.add_camera(res=(640, 480), pos=(1.5, 0.5, 1.0), lookat=(0.5, 0.0, 0.0), fov=40)
scene.build()

end_effector = franka.get_link('hand')
target_quat = np.array([0, 1, 0, 0])
dofs_all = np.arange(9)
neutral_pos = np.array([0.4, 0.0, 0.3]) # 待機（初期）姿勢

# ==========================================
# ＩＫから教師データの生成
# ==========================================
print("【フェーズ１】逆運動学による軌道データを20エピソード分収集します...")
expert_states = []
expert_actions = []
frames_per_episode = 500

def collect_trajectory(s_pos, t_pos):
    global frames_per_episode
    franka.set_qpos(np.zeros(9))
    for _ in range(10): scene.step()

    episode_frames = 0
    def move_to(pos, gripper_open, steps):
        nonlocal episode_frames
        gripper_val = 0.04 if gripper_open else 0.00
        for _ in range(steps):
            current_q = franka.get_qpos().cpu().numpy()[:9]
            state_vector = np.concatenate([current_q, s_pos, t_pos])
            qpos_tensor = franka.inverse_kinematics(link=end_effector, pos=pos, quat=target_quat)
            target_q = qpos_tensor.cpu().numpy().copy()
            target_q[7] = target_q[8] = gripper_val

            expert_states.append(state_vector)
            expert_actions.append(target_q[:9])
            franka.control_dofs_position(target_q[:9], dofs_all)
            scene.step()
            episode_frames += 1

    move_to(np.array([s_pos[0], s_pos[1], 0.20]), True, 60)
    move_to(np.array([s_pos[0], s_pos[1], 0.04]), True, 40)
    move_to(np.array([s_pos[0], s_pos[1], 0.04]), False, 30)
    move_to(np.array([s_pos[0], s_pos[1], 0.20]), False, 40)
    move_to(np.array([t_pos[0], t_pos[1], 0.20]), False, 80)
    move_to(np.array([t_pos[0], t_pos[1], 0.04]), False, 40)
    move_to(np.array([t_pos[0], t_pos[1], 0.04]), True, 30)
    move_to(np.array([t_pos[0], t_pos[1], 0.20]), True, 40)
    move_to(neutral_pos, True, 60) # 次の動作に備えて待機姿勢に戻る

    frames_per_episode = episode_frames

# 20エピソード収集
for ep in range(20):
    noise_s = np.random.uniform(-0.02, 0.02, size=2)
    noise_t = np.random.uniform(-0.02, 0.02, size=2)
    s_pos = np.array([0.5 + noise_s[0], 0.2 + noise_s[1], 0.025])
    t_pos = np.array([0.5 + noise_t[0], -0.2 + noise_t[1], 0.025])
    collect_trajectory(s_pos, t_pos)

# ==========================================
# ＡＩモデルの学習
# ==========================================
print("【フェーズ２】ＡＩの学習を開始します...")
X = torch.tensor(np.array(expert_states), dtype=torch.float32).cuda()
Y = torch.tensor(np.array(expert_actions), dtype=torch.float32).cuda()

class ContextualBehavioralCloning(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 9)
        )
    def forward(self, x):
        return self.net(x)

ai_model = ContextualBehavioralCloning().cuda()
optimizer = optim.Adam(ai_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(1500):
    optimizer.zero_grad()
    loss = loss_fn(ai_model(X), Y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f" Epoch {epoch}: Loss = {loss.item():.6f}")

# ==========================================
# 推論ループ
# ==========================================
print("【フェーズ３】推論による反復タスクを開始します。10回繰り返します...")

# ロボットを初期姿勢にセット
initial_expert_q = expert_states[0][:9]
franka.set_qpos(initial_expert_q)
for _ in range(20): scene.step()

# 録画を開始
cam.start_recording('PAI_final.mp4')

# 現在のスタート地点と目標地点（最初はA→B）
current_start = pos_A
current_target = pos_B

# 安全な静止位置のＩＫを設定
safe_hover_pos = np.array([0.5, 0.0, 0.3])
hover_q_tensor = franka.inverse_kinematics(link=end_effector, pos=safe_hover_pos, quat=target_quat)
hover_q = hover_q_tensor.cpu().numpy().copy()
hover_q[7] = hover_q[8] = 0.04  # 指を開く

# ループを回す
trial_times = 10

for iteration in range(trial_times):
    print(f"試行中... ({iteration + 1}/{trial_times} 回目)")

    # 試行の冒頭で物体の位置をスタート地点にリセット
    try:
        peg_entity.set_pos(current_start)
    except AttributeError:
        peg_entity.set_position(current_start)

    for _ in range(10): # ポーズ
        scene.step()
        cam.render()

    # 往復運搬タスク
    for step in range(frames_per_episode):
        current_q = franka.get_qpos().cpu().numpy()[:9]
        state_vector = np.concatenate([current_q, current_start, current_target])
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).cuda()

        with torch.no_grad():
            inferred_action = ai_model(state_tensor).cpu().numpy()

        franka.control_dofs_position(inferred_action, dofs_all)
        scene.step()
        cam.render()

    # 次の往復のために座標を反転（A→B から B→A へ）
    current_start, current_target = current_target, current_start

    # 試行の間に安全な上空位置に戻して0.5秒間（25フレーム）静止
    for _ in range(25):
        franka.control_dofs_position(hover_q, dofs_all)
        scene.step()
        cam.render()

# 終了前の余韻
for _ in range(50): scene.step(); cam.render()

# 録画終了と保存
cam.stop_recording()

print("録画完了！反復作業を記録した動画『PAI_final.mp4』を保存しました。")

from google.colab import files
import os

# 保存したはずのファイル名
filename = 'PAI_final_True_AI_inference.mp4'

if os.path.exists(filename):
    print(f"{filename} が見つかりました。ローカルへのダウンロードを開始します...")
    files.download(filename)
else:
    print(f"エラー: {filename} が見つかりません。")
    print("左側のフォルダアイコンを開き、別の名前で保存されていないか確認してください。")
