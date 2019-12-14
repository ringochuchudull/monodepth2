This is a modification of original repo 'monodepth2' so that it can be used as an API
The original authors perserves the right to monodepth2

install:
    python setup.py install

Usages:
    encoder, depth_decoder, monoDepth_config, targetDevice = load_mondepthWeight(model_name='mono+stereo_640x192')
    dis, scale_disp, colimg1, pillowim = get_depthDispar(input_image=this_img,
                                model=(encoder, depth_decoder),
                                modelConfig=monoDepth_config,
                                device=targetDevice)


Date of modeification: 14/12/2019