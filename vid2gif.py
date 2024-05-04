import os

vid_folder = 'Videos'
gif_folder = 'Gifs'
if not os.path.exists(gif_folder):
    os.makedirs(gif_folder)

for filename in os.listdir(vid_folder):
    if filename.endswith(".mp4"):
        vid_path = os.path.join(vid_folder, filename)
        gif_path = os.path.join(gif_folder, filename[:-4] + 'as.gif')

        # os.system(f'ffmpeg -i {vid_path} -vf "fps=10,scale=320:-1:flags=lanczos" -c:v gif -loop 0 {gif_path}')
        os.system(f'ffmpeg -i {vid_path} {gif_path}')
        print(f'Converted {vid_path} to {gif_path}')