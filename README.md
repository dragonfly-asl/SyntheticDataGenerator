# SyntheticDataGenerator


## Dependencies
*	ImageMagick (version 7 and higher with C++ bindings)
    -	Debian, Ubuntu: apt-get install imagemagick 
    -	Redhat, Centos: yum install imagemagick
    -	MacOS X: brew install imagemagick

*	PythonMagick (python binding for ImageMagick; tested with 0.9.10 and 0.9.19)
    - can be downloaded and installed by following the instructions on its official ImageMagick github repository: https://github.com/ImageMagick/PythonMagick
    
*	OpenCV (with python bindings; tested with 4.0.1)
*	numpy
    - pip install numpy


## Usage

### Command Line Interface

```
$ python GenerateSyntheticData.py -h
usage: GenerateSyntheticData.py [-h] [-l LOG_LEVEL] -i INPUT -o OUTPUT [-w]
                                [--shift-x SHIFT_X] [--shift-y SHIFT_Y]
                                [--skew-x SKEW_X] [--skew-y SKEW_Y]
                                [--rotate ROTATE] [--horizontal_flip]
                                [--zoom ZOOM] [--contrast CONTRAST]
                                [--brightness BRIGHTNESS]
                                [--saturation SATURATION] [--hue HUE] [--blur]
                                [--blur_radius BLUR_RADIUS]
                                [--blur_sigma BLUR_SIGMA] [--gaussianBlur]
                                [--gaussianBlur_width GAUSSIANBLUR_WIDTH]
                                [--gaussianBlur_sigma GAUSSIANBLUR_SIGMA]
                                [--despeckle] [--enhance] [--equalize]
                                [--gamma GAMMA] [--implode IMPLODE] [--negate]
                                [--normalize] [--quantize]
                                [--reduceNoise REDUCENOISE] [--shade]
                                [--shade_azimuth SHADE_AZIMUTH]
                                [--shade_elevation SHADE_ELEVATION]
                                [--sharpen] [--sharpen_radius SHARPEN_RADIUS]
                                [--sharpen_sigma SHARPEN_SIGMA]
                                [--swirl SWIRL] [--wave]
                                [--wave_amplitude WAVE_AMPLITUDE]
                                [--wave_wavelength WAVE_WAVELENGTH] [--auto]
                                [--auto_ops AUTO_OPS]
                                [--auto_rotate_min AUTO_ROTATE_MIN]
                                [--auto_rotate_max AUTO_ROTATE_MAX]
                                [--auto_zoom_min AUTO_ZOOM_MIN]
                                [--auto_zoom_max AUTO_ZOOM_MAX]

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        log-level (INFO|WARN|DEBUG|FATAL|ERROR)
  -i INPUT, --input INPUT
                        Input image file name
  -o OUTPUT, --output OUTPUT
                        Output image file name
  -w, --overwrite       If set, will overwrite the existing output file
  --shift-x SHIFT_X
  --shift-y SHIFT_Y
  --skew-x SKEW_X
  --skew-y SKEW_Y
  --rotate ROTATE       rotates image clock- or counterclock-wise (angle in
                        degrees)
  --horizontal_flip     horizontally flips image
  --zoom ZOOM           resize image; argument given in percentage
  --contrast CONTRAST   default=0; 0~infinity (integer times contract is
                        applided to image)
  --brightness BRIGHTNESS
                        default=100
  --saturation SATURATION
                        default=100
  --hue HUE             default=100
  --blur
  --blur_radius BLUR_RADIUS
  --blur_sigma BLUR_SIGMA
  --gaussianBlur
  --gaussianBlur_width GAUSSIANBLUR_WIDTH
  --gaussianBlur_sigma GAUSSIANBLUR_SIGMA
  --despeckle
  --enhance
  --equalize
  --gamma GAMMA         0 ~ 2; 1 is default
  --implode IMPLODE     Implode factor 0~1; 0 (nothing) to 1 (full); 0.0 ~ 0.5
                        recommended.
  --negate
  --normalize
  --quantize
  --reduceNoise REDUCENOISE
                        default=1
  --shade
  --shade_azimuth SHADE_AZIMUTH
  --shade_elevation SHADE_ELEVATION
  --sharpen
  --sharpen_radius SHARPEN_RADIUS
  --sharpen_sigma SHARPEN_SIGMA
  --swirl SWIRL         degree; default=10
  --wave
  --wave_amplitude WAVE_AMPLITUDE
  --wave_wavelength WAVE_WAVELENGTH
  --auto
  --auto_ops AUTO_OPS
  --auto_rotate_min AUTO_ROTATE_MIN
  --auto_rotate_max AUTO_ROTATE_MAX
  --auto_zoom_min AUTO_ZOOM_MIN
  --auto_zoom_max AUTO_ZOOM_MAX
$
```

### Generating a synthetic video/image using a set of randomly selected options
```
$ python GenerateSyntheticData.py -i video.mov -o video-synthetic-auto.mov --auto
INFO:DragonFly-ASL-GSD:Random options: Namespace(blur=None, brightness=100.76630360332423, contrast=0, despeckle=False, enhance=True, equalize=None, gamma=1.0414632664690044, gaussianBlur=None, horizontal_flip=None, hue=103.42793104962132, implode=0, negate=None, normalize=False, quantize=True, reduceNoise=0, rotate=0, saturation=95.07533831166259, shade=False, sharpen=False, shift_x=2, shift_y=0, skew_x=0, skew_y=0, swirl=-0.19551897200245116, wave=None, zoom='97%')
```


### Manipulating an input video/image using an explicitly chosen operation(s) and parameters

Zoom 90%:
```
python GenerateSyntheticData.py --zoom 90% -i video.mov -o video-synthetic-zoom90%.mov
```

Rotate 24 degrees clock-wise:
```
python GenerateSyntheticData.py --rotate 24 -i video.mov -o video-synthetic-rotate24.mov
```
