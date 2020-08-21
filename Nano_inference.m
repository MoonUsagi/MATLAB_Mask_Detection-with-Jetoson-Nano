hwobj= jetson('192.168.51.155','usagi','0000');
%hwobj= jetson('192.168.43.77','usagi','0000');

envCfg = coder.gpuEnvConfig('jetson');
envCfg.BasicCodegen = 1;
envCfg.BasicCodeexec = 1;
%envCfg.DeepLibTarget = 'tensorrt';
envCfg.DeepLibTarget = 'cudnn';
envCfg.DeepCodegen = 1;
envCfg.DeepCodeexec = 1;
envCfg.Quiet = 1;
envCfg.HardwareObject = hwobj;
coder.checkGpuInstall(envCfg);
%%
type inference_mask_multi
 
cfg = coder.gpuConfig('exe');
cfg.Hardware = coder.hardware('NVIDIA Jetson');
%cfg.Hardware.BuildDir = '~/remoteBuildTest';
cfg.GenerateExampleMain = 'GenerateCodeAndCompile'

codegen('-config ', cfg,'inference_mask_multi', '-report');