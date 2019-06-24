# /bin/env python
# coding: utf-8

from __future__ import print_function

import sys
import argparse
import logging
import os
import math
import cv2
import numpy as np


class GenerateSyntheticData:

    import PythonMagick as Magick

    def __init__(self, logger=None):
        if logger == None:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            self.logger = logging.getLogger()
        else:
            self.logger = logger

    @staticmethod
    def appendArgumentParser(argparser):
        argparser.add_argument('--shift-x', type=int, help='')
        argparser.add_argument('--shift-y', type=int, help='')
        argparser.add_argument('--skew-x', type=float, help='')
        argparser.add_argument('--skew-y', type=float, help='')
        argparser.add_argument('--rotate', type=float, help='rotates image clock- or counterclock-wise (angle in degrees)')
        argparser.add_argument('--horizontal_flip', action='store_true', help='horizontally flips image')
        
        argparser.add_argument('--zoom', type=str, help='resize image; argument given in percentage')
        
        argparser.add_argument('--contrast', type=int, help='default=0; 0~infinity (integer times contract is applided to image)')
        
        argparser.add_argument('--brightness', type=float, help='default=100')
        argparser.add_argument('--saturation', type=float, help='default=100')
        argparser.add_argument('--hue', type=float, help='default=100')
        
        argparser.add_argument('--blur', action='store_true', help='')
        argparser.add_argument('--blur_radius', type=float, default=10, help='')
        argparser.add_argument('--blur_sigma', type=float, default=1, help='')
        
        argparser.add_argument('--gaussianBlur', action='store_true', help='')
        argparser.add_argument('--gaussianBlur_width', type=float, default=5, help='')
        argparser.add_argument('--gaussianBlur_sigma', type=float, default=1, help='')
        
        argparser.add_argument('--despeckle', action='store_true', help='')
        argparser.add_argument('--enhance', action='store_true', help='')
        argparser.add_argument('--equalize', action='store_true', help='')
        
        argparser.add_argument('--gamma', type=float, help='0 ~ 2; 1 is default')
        
        argparser.add_argument('--implode', type=float, help='Implode factor 0~1; 0 (nothing) to 1 (full); 0.0 ~ 0.5 recommended.')
        
        argparser.add_argument('--negate', action='store_true', help='')
        argparser.add_argument('--normalize', action='store_true', help='')
        argparser.add_argument('--quantize', action='store_true', help='')
        
        argparser.add_argument('--reduceNoise', type=int, help='default=1')
        
        argparser.add_argument('--shade', action='store_true', help='')
        argparser.add_argument('--shade_azimuth', type=float, default=50, help='')
        argparser.add_argument('--shade_elevation', type=float, default=50, help='')
        
        argparser.add_argument('--sharpen', action='store_true', help='')
        argparser.add_argument('--sharpen_radius', type=float, default=1, help='')
        argparser.add_argument('--sharpen_sigma', type=float, default=0.5, help='')
        
        argparser.add_argument('--swirl', type=float, help='degree; default=10')
        
        argparser.add_argument('--wave', action='store_true', help='')
        argparser.add_argument('--wave_amplitude', type=float, default=5, help='')
        argparser.add_argument('--wave_wavelength', type=float, default=100, help='')

        argparser.add_argument('--auto', action='store_true', help='')

        argparser.add_argument('--auto_ops', type=str, default='', help='')

        argparser.add_argument('--auto_rotate_min', type=float, default=0, help='')
        argparser.add_argument('--auto_rotate_max', type=float, default=0, help='')

        argparser.add_argument('--auto_zoom_min', type=float, default=0, help='')
        argparser.add_argument('--auto_zoom_max', type=float, default=0, help='')


    def generateRandomOptions(self, cmdArg):

        def _generateRandomOptionsShift(args):
            args.shift_x = int(np.abs(np.random.normal(0, 3)))  # -10 ~ +10
            args.shift_y = int(np.abs(np.random.normal(0, 1)))  # -3 ~ +3

        def _generateRandomOptionsSkew(args):
            args.skew_x = int(np.random.normal(0, 3))  # -10 ~ +10
            args.skew_y = int(np.random.normal(0, 3))  # -10 ~ +10

        def _generateRandomOptionsRotate(args):
            if cmdArg.auto_rotate_min != cmdArg.auto_rotate_max:
                args.rotate = int(np.random.uniform(cmdArg.auto_rotate_min, cmdArg.auto_rotate_max))
            else:
                args.rotate = int(np.random.normal(0, 3))  # -10 ~ +10

        def _generateRandomOptionsZoom(args):
            if cmdArg.auto_zoom_min != cmdArg.auto_zoom_max:
                args.zoom = str(int(np.random.uniform(cmdArg.auto_zoom_min, cmdArg.auto_zoom_max))) + '%'
            else:
                args.zoom = str(int(np.random.normal(100, 3))) + '%'  # 90% ~ 110%

        def _generateRandomOptionsContrast(args):
            args.contrast = int(np.abs(np.random.normal(0, 1)))  # 0 ~ +3

        def _generateRandomOptionsBrightness(args):
                args.brightness = np.random.normal(100, 5)  # 85 ~ 115

        def _generateRandomOptionsSaturation(args):
                args.saturation = np.random.normal(100, 5)  # 85 ~ 115

        def _generateRandomOptionsHue(args):
            args.hue = np.random.normal(100, 5)  # 85 ~ 115

        def _generateRandomOptionsBlur(args):
            if np.random.binomial(1,0.1):  # do blur
                if np.random.binomial(1,0.5):
                    args.blur = True
                else:
                    args.gaussianBlur = True
            if args.blur:
                args.blur_radius = np.abs(np.random.normal(0, 3))  # 0 ~ 10
                args.blur_sigma = np.abs(np.random.normal(0, 0.7))  # 0 ~ 2
            if args.gaussianBlur:
                args.gaussianBlur_width = np.abs(np.random.normal(0, 3))  # 0 ~ 10
                args.gaussianBlur_sigma = np.abs(np.random.normal(0, 0.7))  # 0 ~ 2

        def _generateRandomOptionsHorizontalFlip(args):
            args.horizontal_flip = (np.random.binomial(1,0.1) > 0)

        def _generateRandomOptionsDespeckle(args):
            args.despeckle = (np.random.binomial(1,0.5) > 0)

        def _generateRandomOptionsEnhance(args):
            args.enhance = (np.random.binomial(1,0.5) > 0)

        def _generateRandomOptionsEqualize(args):
            args.equalize = (np.random.binomial(1,0.1) == 1)

        def _generateRandomOptionsNegate(args):
            args.negate = (np.random.binomial(1,0.1) == 1)

        def _generateRandomOptionsNormalize(args):
                args.normalize = (np.random.binomial(1,0.1) > 0)

        def _generateRandomOptionsQuantize(args):
            args.quantize = (np.random.binomial(1,0.1) > 0)

        def _generateRandomOptionsGamma(args):
            args.gamma = np.abs(np.random.normal(1, 0.03))  # 0 ~ 2

        def _generateRandomOptionsImplode(args):
            args.implode = 0
            if np.random.binomial(1,0.5) > 0:
                args.implode = np.random.normal(0, 0.15)  # -0.5 ~ 0.5

        def _generateRandomOptionsReduceNoise(args):
            args.reduceNoise = int(np.abs(np.random.normal(0, 0.7)))  # 0 ~ 2

        def _generateRandomOptionsShade(args):
            args.shade = (np.random.binomial(1,0.1) > 0)
            if args.shade:
                args.shade_azimuth = np.random.normal(50, 17)  # 0 ~ 100
                args.shade_elevation = np.random.normal(50, 17)  # 0 ~ 100

        def _generateRandomOptionsSharpen(args):
            args.sharpen = (np.random.binomial(1,0.1) > 0)
            if args.sharpen:
                args.sharpen_radius = np.abs(np.random.normal(0, 0.7))  # 0 ~ 2
                args.sharpen_sigma = np.abs(np.random.normal(0, 0.3))  # 0 ~ 1

        def _generateRandomOptionsSwirl(args):
            args.swirl = np.random.normal(0, 5)  # -15 ~ +15

        def _generateRandomOptionsWave(args):
            args.wave = (np.random.binomial(1,0.3) > 0)
            if args.wave:
                args.wave_amplitude = np.abs(np.random.normal(5, 0.3))  # 0 ~ 10
                args.wave_wavelength = np.abs(np.random.normal(100, 10))  # 0 ~ 200

        args = argparse.Namespace()
        args.shift_x = args.shift_y = None
        args.skew_x = args.skew_y = None
        args.rotate = args.zoom = None
        args.contrast = args.brightness = args.saturation = args.hue = None
        args.blur = args.gaussianBlur = None
        args.horizontal_flip = None
        args.despeckle = args.enhance = args.reduceNoise = None
        args.equalize = args.negate = args.normalize = args.quantize = args.gamma = None
        args.shade = None
        args.sharpen = None
        args.implode = args.swirl = args.wave = None

        if len(cmdArg.auto_ops)>0:
            for op in cmdArg.auto_ops.split(","):
                if op == 'shift':               _generateRandomOptionsShift(args)
                elif op == 'skew':              _generateRandomOptionsSkew(args)
                elif op == 'rotate':            _generateRandomOptionsRotate(args)
                elif op == 'zoom':              _generateRandomOptionsZoom(args)
                elif op == 'contrast':          _generateRandomOptionsContrast(args)
                elif op == 'brightness':        _generateRandomOptionsBrightness(args)
                elif op == 'saturation':        _generateRandomOptionsSaturation(args)
                elif op == 'hue':               _generateRandomOptionsHue(args)
                elif op == 'blur':              _generateRandomOptionsBlur(args)
                elif op == 'horizontal_flip':   _generateRandomOptionsHorizontalFlip(args)
                elif op == 'despeckle':         _generateRandomOptionsDespeckle(args)
                elif op == 'enhance':           _generateRandomOptionsEnhance(args)
                elif op == 'equalize':          _generateRandomOptionsEqualize(args)
                elif op == 'negate':            _generateRandomOptionsNegate(args)
                elif op == 'normalize':         _generateRandomOptionsNormalize(args)
                elif op == 'quantize':          _generateRandomOptionsQuantize(args)
                elif op == 'gamma':             _generateRandomOptionsGamma(args)
                elif op == 'implode':           _generateRandomOptionsImplode(args)
                elif op == 'reduceNoise':       _generateRandomOptionsReduceNoise(args)
                elif op == 'shade':             _generateRandomOptionsShade(args)
                elif op == 'sharpen':           _generateRandomOptionsSharpen(args)
                elif op == 'swirl':             _generateRandomOptionsSwirl(args)
                elif op == 'wave':              _generateRandomOptionsWave(args)
                else:
                    self.logger.error('Unknown Operation Name ' + op)

        else: # apply all operations
            _generateRandomOptionsShift(args)
            _generateRandomOptionsSkew(args)
            _generateRandomOptionsRotate(args)
            _generateRandomOptionsZoom(args)
            _generateRandomOptionsContrast(args)
            _generateRandomOptionsBrightness(args)
            _generateRandomOptionsSaturation(args)
            _generateRandomOptionsHue(args)
            _generateRandomOptionsBlur(args)
            #_generateRandomOptionsHorizontalFlip(args)
            _generateRandomOptionsDespeckle(args)
            _generateRandomOptionsEnhance(args)
            #_generateRandomOptionsEqualize(args)
            #_generateRandomOptionsNegate(args)
            _generateRandomOptionsNormalize(args)
            _generateRandomOptionsQuantize(args)
            _generateRandomOptionsGamma(args)
            _generateRandomOptionsImplode(args)
            _generateRandomOptionsReduceNoise(args)
            _generateRandomOptionsShade(args)
            _generateRandomOptionsSharpen(args)
            _generateRandomOptionsSwirl(args)
            #_generateRandomOptionsWave(args)

        self.logger.debug('Randomly generated options: ')
        for key in vars(args):
            self.logger.debug(' -- %s: %s' % (key, getattr(args, key)))
        self.logger.debug('')

        return args

    def isVideo(self, inputF):
        video_file_extensions = (
            '.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
            '.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
            '.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
            '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
            '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
            '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
            '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
            '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
            '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
            '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
            '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
            '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
            '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
            '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
            '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
            '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
            '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
            '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
            '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
            '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
            '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
            '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
            '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
            '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg', '.vem', '.vep', '.vf', '.vft',
            '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
            '.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
            '.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
            '.zm1', '.zm2', '.zm3', '.zmv')
        if inputF.endswith((video_file_extensions)):
            return True
        return False

    def getFPS(self, vF):
        video = cv2.VideoCapture(vF);
        major_ver, _, _ = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps

    def splitFromVideo(self, inputF, outputFPrefix):
        retVal = []
        vid = cv2.VideoCapture(inputF)
        idx = 0
        while(True):
            ret, frame = vid.read()
            if not ret:
                break
            name = outputFPrefix + '_frame' + str(idx) + '.png'
            cv2.imwrite(name, frame)
            retVal.append(name)
            idx += 1
        return retVal

    def mergeIntoVideo(self, inFs, outputF, FPS):
        frame = cv2.imread(inFs[0])
        height, width, _ = frame.shape
        video = cv2.VideoWriter(outputF, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
        for inF in inFs:
            video.write(cv2.imread(inF))
        video.release()

    def generate(self, inputF, outputF, args):

        if args.auto:
            auto_options = self.generateRandomOptions(args)
            logger.info('Random options: ' + str(auto_options))

        if self.isVideo(inputF):
            FPS = self.getFPS(inputF)
            inputFs = self.splitFromVideo(inputF, outputF+'_input')
            outputFs = []

            for idx in range(0, len(inputFs)):
                iF = inputFs[idx]
                oF = outputF + '_output_frame' + str(idx) + '.png'
                if args.auto:
                    self._generate(iF, oF, auto_options)
                else:
                    self._generate(iF, oF, args)
                outputFs.append(oF)
            self.mergeIntoVideo(outputFs, outputF, FPS)

            for f in inputFs:
                os.remove(f)
            for f in outputFs:
                os.remove(f)

            return True

        else:
            if args.auto:
                return self._generate(inputF, outputF, auto_options)
            else:
                return self._generate(inputF, outputF, args)


    def _generate(self, inputF, outputF, args):

        inputImage = self.Magick.Image(inputF)
    
        input_width = inputImage.size().width()
        input_height = inputImage.size().height()
    
        self.logger.debug('Input width and height: %d x %d' % (input_width, input_height))
    
        # make image ready to be modified
        inputImage.modifyImage()
        inputImage.backgroundColor(self.Magick.Color('black'))
    
        if args.shift_x != None:
            inputImage.roll(args.shift_x, 0)
        if args.shift_y != None:
            inputImage.roll(0, args.shift_y)
    
        if args.skew_x != None and args.skew_y != None:
            inputImage.shear(args.skew_x, args.skew_y)
        elif args.skew_x != None:
            inputImage.shear(args.skew_x, 0)
        if args.skew_y != None:
            inputImage.shear(0, args.skew_y)

        if args.rotate != None:
            inputImage.rotate(args.rotate)
            inputImage.crop(self.Magick.Geometry(input_width, input_height, 0, 0))

        if args.horizontal_flip:
            inputImage.flop()

        if args.zoom != None:
            inputImage.sample(self.Magick.Geometry(args.zoom))
            if int(args.zoom.strip()[0:-1]) >= 100:
                inputImage.crop(self.Magick.Geometry(input_width,
                                                 input_height,
                                                 int((inputImage.size().width() - input_width) / 2),
                                                 int((inputImage.size().height() - input_height) / 2)))
            else:
                # PythonMagick is missing extent() API
                # inputImage.exent(Magick.Geometry(input_width, input_height), Magick.GravityType.CenterGravity)
                smallWidth = inputImage.size().width()
                smallHeight = inputImage.size().height()
                inputImage.size(self.Magick.Geometry(input_width, input_height))
                inputImage.draw(self.Magick.DrawableRectangle(smallWidth, smallHeight, input_width, input_height))
                inputImage.draw(self.Magick.DrawableRectangle(smallWidth, 0, input_width, smallHeight))
                inputImage.draw(self.Magick.DrawableRectangle(0, smallHeight, smallWidth, input_height))
                inputImage.roll(int((input_width - smallWidth) / 2), int((input_height - smallHeight) / 2))

        if args.contrast != None:
            for _ in range(0, args.contrast):
                inputImage.contrast(args.contrast)
    
        if args.brightness != None or args.saturation != None or args.hue != None:
            if args.brightness is None:
                args.brightness = 100
            if args.saturation is None:
                args.saturation = 100
            if args.hue is None:
                args.hue = 100
            inputImage.modulate(args.brightness, args.saturation, args.hue) 
    
        if args.blur:
            inputImage.blur(args.blur_radius, args.blur_sigma)
    
        if args.gaussianBlur:
            inputImage.gaussianBlur(args.gaussianBlur_width, args.gaussianBlur_sigma)
    
        if args.despeckle:
            inputImage.despeckle()
        if args.enhance:
            inputImage.enhance()
        if args.equalize:
            inputImage.equalize()
        if args.gamma != None:
            inputImage.gamma(args.gamma)
    
        if args.implode != None:
            inputImage.implode(args.implode)
    
        if args.negate:
            inputImage.negate()
        if args.normalize:
            inputImage.normalize()
        if args.quantize:
            inputImage.quantize()
        if args.reduceNoise != None:
            inputImage.reduceNoise(args.reduceNoise)
    
        if args.shade:
            inputImage.shade(args.shade_azimuth, args.shade_elevation)
        if args.sharpen:
            inputImage.sharpen(args.sharpen_radius, args.sharpen_sigma)
    
        if args.swirl != None:
            inputImage.swirl(args.swirl)
    
        if args.wave:
            inputImage.wave(args.wave_amplitude, args.wave_wavelength)
            inputImage.crop(self.Magick.Geometry(input_width,
                                                 input_height,
                                                 int(math.fabs((inputImage.size().width() - input_width) / 2)),
                                                 int(math.fabs((inputImage.size().height() - input_height) / 2))))
    
        inputImage.write(outputF)
        self.logger.debug('Output width and height: %d x %d' % (inputImage.size().width(), inputImage.size().height()))

        return True


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-l', '--log-level', default='INFO', help="log-level (INFO|WARN|DEBUG|FATAL|ERROR)")
    argparser.add_argument('-i', '--input', required=True, help='Input image file name')
    argparser.add_argument('-o', '--output', required=True, help='Output image file name')
    argparser.add_argument('-w', '--overwrite', action='store_true', help='If set, will overwrite the existing output file')
    GenerateSyntheticData.appendArgumentParser(argparser)
    args = argparser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=args.log_level)
    logger = logging.getLogger("DragonFly-ASL-GSD")

    logger.debug('CLI arguments')
    for key in vars(args):
        logger.debug(' -- %s: %s' % (key, getattr(args, key)))
    logger.debug('')

    # check input file exists
    if not os.path.isfile(args.input):
        logger.error('Input file %s does not exist: ' % args.input)
        sys.exit(1)
    # check if output file exists
    if os.path.isfile(args.output) and not args.overwrite:
        try: input = raw_input
        except NameError: pass
        yn = input('Do you wish to overwrite %s? (y/n) ' % args.output)
        if yn != 'y' and yn != 'Y':
            logger.error('Output file %s will not be overwritten.' % args.output)
            sys.exit(1)

    GSD = GenerateSyntheticData(logger=logger)
    status = GSD.generate(args.input, args.output, args)
    logger.debug('Generation status: %r' % status)

