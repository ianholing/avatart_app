import torch
import torch
import torchvision
from PIL import Image, ImageDraw
from tools.face_features import extract_face_features, angle_between, point_after_rotation
import cv2
from tqdm.auto import tqdm
import numpy as np
from IPython.display import Video
import subprocess

# JSON TASK LOADER
import sys, os, json, shutil

# DIRECTORY MONITOR
from watchdog.events import RegexMatchingEventHandler
from watchdog.observers import Observer
import time

# GIGATRONIC INFERENCE
from MEGATRONIC.megamask import megamask
from MEGATRONIC.gigatronic import gigatronic
from MEGATRONIC.mask_utils import translate_mask
from glob import glob

MODEL = "MEGATRONIC/gigatronic-drawings-finetuned-v0.pt"
MASK_MODEL = "MEGATRONIC/megamask_v2-v0.pt"
STYLES = glob("used_styles/*")

def load_image(path, device):
    return transforms(Image.open(path).convert("RGB")).to(device)

def load_new_style(path, device):
    style = load_image(path, device).unsqueeze(0)

    with torch.no_grad():
        style_mask = megamask(style).div(0.01).softmax(1)
        style_mask = (style_mask > 0.5).float()
        style_encodings = gigatronic.encode_styles(style, style_mask)

    return style_encodings

def find_crop_parameters(image, outer_spacing = 0.38):
    face_features = extract_face_features(np.array(image))
    
    if len(face_features) > 0:
        features = face_features[0]
        
        le = features['left_eye']
        lec = le[:,0].mean(), le[:,1].mean()

        re = features['right_eye']
        rec = re[:,0].mean(), re[:,1].mean()

        m  = features['mouth']
        mc = m[:,0].mean(), m[:,1].mean()

        angle = angle_between(lec, rec)

        lec = point_after_rotation(lec, angle, image.width, image.height)
        rec = point_after_rotation(rec, angle, image.width, image.height)

        cy = (lec[1] + rec[1]) / 2
        cx = (lec[0] + rec[0]) / 2
        dx = abs(rec[0] - lec[0])
        sx = ((dx * (outer_spacing*2) / (1-outer_spacing*2)) / 2) + dx / 2
        syu = sx*2 * outer_spacing
        syd = sx*2 * (1-outer_spacing)
        
        return angle, (cx, cy), (sx, syu, syd)
        
    else:
        raise Exception('Couldn\'t find any face in the provided image!')
        
def get_croped_frame(image, angle, c, s, size=256):
    cx, cy = c
    sx, syu, syd = s
    
    face = image.copy()
    face = face.rotate(angle)
    draw = ImageDraw.Draw(face)
    
    face = face.crop([cx-sx, cy-syu, cx+sx, cy+syd])
    face = face.resize((size, size))
    face = torch.from_numpy(np.array(face))  
    return face

def crop_video(frames, crop_reference_frame=0, size=256, outer_spacing=0.38):
    cropped_frames = []
    # Find crop parameters
    ref_frame = Image.fromarray(frames[crop_reference_frame].numpy())

    angle, c, s = find_crop_parameters(ref_frame, outer_spacing)
    
    for frame in tqdm(frames):
        image = Image.fromarray(frame.numpy())
        face = get_croped_frame(image, angle, c, s, size)
        cropped_frames.append(face)
        
    return torch.stack(cropped_frames, dim=0).permute(0, 3, 1, 2).float() / 255


##################################################################

class ImagesWatcher:
    def __init__(self, src_path):
        self.__src_path = src_path
        self.__event_handler = ImagesEventHandler()
        self.__event_observer = Observer()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop()

    def start(self):
        self.__schedule()
        self.__event_observer.start()

    def stop(self):
        self.__event_observer.stop()
        self.__event_observer.join()

    def __schedule(self):
        self.__event_observer.schedule(
            self.__event_handler,
            self.__src_path,
            recursive=False
        )

##################################################################
        
class ImagesEventHandler(RegexMatchingEventHandler):
    THUMBNAIL_SIZE = (128, 128)
    IMAGES_REGEX = [r".*\.json$"]

    def __init__(self):
        super().__init__(self.IMAGES_REGEX)

    def on_created(self, event):
        self.process(event)

    def process(self, event):
        filename, ext = os.path.splitext(event.src_path)
        print ("Filename created: ", filename, ext);
        
        if not os.path.isfile(filename + ext):
            print ('File not exist:', filename, ext)
            return
        
        try:
            print ("LOAD TASK DATA")
            with open(filename + ext) as json_file:
                data = json.load(json_file)

                uid = data['uid']
                file = data['file']
                style = data['style']

            print("TASK IN:", filename + ext, " == { UID:", uid, ", Style:", style, ", File:", file, '}')

            ## TAKE CARE OF EXIF:
            final_file = './uploads/'+file
            if not file.endswith('.avi'):
                final_file = './uploads/'+uid+".avi"
                print("EXIF PROCESS MP4 TO AVI:", final_file)
                subprocess.call(['ffmpeg', '-y', '-i', './uploads/'+file, '-an', final_file])

            print ("LOAD VIDEO PROCESS")
            try:
                frames, _, info = torchvision.io.read_video(final_file, start_pts=0, end_pts=None, pts_unit='pts')
                print("LOADED FRAMES: ", frames.shape)
            except:
                print ("Unexpected error:", sys.exc_info()[0])
                json_data = { 'file': filename + ext, 'error': 'Error loading video'}
                with open('./errors/'+str(uid)+'.json', 'w') as fp:
                    json.dump(json_data, fp)
                return

            print ("CROP PROCESS")
            try:
                cropped_frames = crop_video(frames, outer_spacing=0.38)
                print("CROPPED FRAMES: ", cropped_frames.shape)
            except:
                print ("Unexpected error:", sys.exc_info()[0])
                json_data = { 'file': filename + ext, 'error': 'Error cropping, face not detected'}
                with open('./errors/'+str(uid)+'.json', 'w') as fp:
                    json.dump(json_data, fp)
                return

            cropped_file = 'cropped/' + uid + '.mp4'
            output_frames = (cropped_frames * 255).permute(0, 2, 3, 1).long()
#             torchvision.io.write_video('processed/' + uid + '_' + style+'.mp4', output_frames, fps=info['video_fps'])

            print ("EXTRACT LATENTS")
            with torch.no_grad():
                all_mask_encodings = []
                all_style_encodings = []
                batch_size=4
                for i in tqdm(range(0, cropped_frames.size(0), batch_size)):
                    masks = megamask(cropped_frames[i:i+batch_size].to(device)).div(0.01).softmax(1)
                    masks = (masks > 0.5).float()

                    mask_encodings = gigatronic.encode_mask(masks)
                    style_encodings = gigatronic.encode_styles(cropped_frames[i:i+batch_size].to(device), masks)

                    all_mask_encodings.append(mask_encodings)
                    all_style_encodings.append(style_encodings)

                all_mask_encodings = torch.cat(all_mask_encodings, dim=0)
                all_style_encodings = [torch.cat([e[i] for e in all_style_encodings]) for i in range(8)]

            print("MASKS SHAPE:", all_mask_encodings.shape, len(all_style_encodings), all_style_encodings[0].shape)

            print ("APPLY A CUSTOM STYLE", style)
            theta = 0.9
            selected_style = int(style)-1
            updated_style_encodings = []
            for i in range(len(all_style_encodings)):
                updated_style_encodings.append((style_embedding[selected_style][i] * theta) + (all_style_encodings[i] * (1-theta)))

            print ("RECONSTRUCT USING SELF LATENTS")
            with torch.no_grad():
                all_gen_frames = []
                batch_size=4
                for i in tqdm(range(0, frames.size(0), batch_size)):
                    mask_encodings = all_mask_encodings[i:i+batch_size]
                    style_encodings = [ase[i:i+batch_size] for ase in updated_style_encodings]
                    gen_frames = gigatronic.decode(mask_encodings, style_encodings)
                    all_gen_frames.append(gen_frames)

                all_gen_frames = torch.cat(all_gen_frames, dim=0)

            print("GENERATED SHAPE:", all_gen_frames.shape)

            all_gen_frames = (all_gen_frames * (1 - watermark[:, -1])) + watermark[:, :-1]
            output_frames = (all_gen_frames * 255).permute(0, 2, 3, 1).long()
            torchvision.io.write_video('static/processed/' + uid + '_' + style+'_video.mp4', output_frames, fps=info['video_fps'])
            subprocess.call(['ffmpeg', '-y', '-i', './uploads/'+file, 'static/processed/' + uid + '_' + style+'_audio_org.mp3'])
            subprocess.call(['ffmpeg', '-y', '-i', 'static/processed/' + uid + '_' + style+'_audio_org.mp3', '-af', 'asetrate=44100*0.9,aresample=44100,atempo=1.2', '-vol', '400', 'static/processed/' + uid + '_' + style+'_audio_down.mp3'])
            subprocess.call(['ffmpeg', '-y', '-i', 'static/processed/' + uid + '_' + style+'_audio_org.mp3', '-i', 'static/processed/' + uid + '_' + style+'_audio_down.mp3', '-filter_complex', 'amix=inputs=2:duration=longest', 'static/processed/' + uid + '_' + style+'_audio.mp3'])
            subprocess.call(['ffmpeg', '-y', '-i', 'static/processed/' + uid + '_' + style+'_video.mp4', '-i', 'static/processed/' + uid + '_' + style+'_audio.mp3', '-c:v', 'copy', '-c:a', 'aac', 'static/processed/' + uid + '_' + style+'.mp4'])

            
        except:
            print ("Unexpected error:", sys.exc_info())
            json_data = { 'file': filename + ext, 'error': 'Unknown'}
            with open('./errors/'+str(uid)+'.json', 'w') as fp:
                json.dump(json_data, fp)
            return

        finally:
            #TODO: UID MAY NOT EXIST
            shutil.move(filename + ext, "static/processed/"+uid+'_'+style+ext)
        

##################################################################

print("LOADING PYTORCH..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

watermark = torchvision.transforms.ToTensor()(Image.open("Watermark.png")).unsqueeze(0).to(device)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor()
])

print ("LOADING GIGATRONIC..")
gigatronic.load_state_dict(torch.load(MODEL, map_location=torch.device(device)))
gigatronic.to(device)
gigatronic.eval()

print ("LOADING MEGAMASK..")
megamask.load_state_dict(torch.load(MASK_MODEL, map_location=torch.device(device)))
megamask.to(device)
megamask.eval()

print ("LOADING STYLES..")
style_embedding = [load_new_style(s, device) for s in STYLES]
print (type(style_embedding))
print (type(style_embedding[0]))
print (len(style_embedding))
print (len(style_embedding[0]))
print (style_embedding[0][0].shape)
print("-----")

if __name__ == '__main__':
    src_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    ImagesWatcher(src_path).run()
    
