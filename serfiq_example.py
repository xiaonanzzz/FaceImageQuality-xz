# Author: Jan Niklas Kolf, 2020
from face_image_quality import SER_FIQ
import cv2
import os
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Sample code of calculating the score of an image
    
    # Create the SER-FIQ Model
    # Choose the GPU, default is 0.
    ser_fiq = SER_FIQ(gpu=None)
        
    # Load the test image
    imgdir = 'Imagens_Gol'
    outdir = 'Imagens_Gol_out'
    df = []
    Path(outdir).mkdir(exist_ok=True)

    for x in Path(imgdir).iterdir():
        if x.is_file() and x.suffix in ['.jpg', '.png', '.jpeg']:
            print(x)
            imgf = str(x)
            test_img = cv2.imread(imgf)
            print('shape', test_img.shape)

        
            # Align the image
            aligned_img = ser_fiq.apply_mtcnn(test_img)
            print(aligned_img.shape if aligned_img is not None else 'None')
            
            # Calculate the quality score of the image
            # T=100 (default) is a good choice
            # Alpha and r parameters can be used to scale your
            # score distribution.
            if aligned_img is None:
                score = 0
            else:
                score = ser_fiq.get_score(aligned_img, T=100)
            
            h,w = test_img.shape[:2]
            cv2.putText(test_img, 'Quality: {:.4f}'.format(score), (int(w*0.25), 100), cv2.FONT_HERSHEY_COMPLEX, 1.5 if (h < 2000) else 2.5, (0, 0, 255), 2)
            cv2.imwrite(str(Path(outdir) / x.name), test_img)
            print(x, score)

            df.append(dict(image=x, score=score))
    df = pd.DataFrame(df)
    df = df.sort_values('score')
    df.to_csv(outdir + '/FaceImageQuality-scores.csv', index=False)
