function augment_iq_5perturbs_keepSNR()
% 五扰动：CFO,SCALE,GAIN,SHIFT,CHIRP；保持原信号均方功率；新增按SNR加复高斯白噪声
% 输出：与旧版完全兼容（I/Q字段、Z、S、fs、perturb_order、disturb_type/params）
% 新增：snr_db[Nx1]、noise_var[Nx1]（不改变Z/S的形状与含义）
clc; clear; close all;

[fn, fp] = uigetfile({'*.mat','MAT-files (*.mat)'}, '选择输入 MAT 文件'); 
if isequal(fn,0), return; end
infile = fullfile(fp, fn);
[outfn, outfp] = uiputfile('*.mat','选择保存名', fullfile(fp, [erase(fn,'.mat') '_aug5_keepSNR.mat']));
if isequal(outfn,0), return; end
outfile = fullfile(outfp, outfn);

S = load(infile); fields = fieldnames(S);

% ---------------- 采样率 ----------------
fs = []; 
for k = ["fs","Fs","FS","sample_rate","samp_rate","sampling_rate","sampling_frequency"]
    if isfield(S,k), fs = double(S.(k)); break; end
end
if isempty(fs), fs = 50e6; fprintf('未找到fs，默认 %.1f Hz\n', fs); end

% ---------------- I/Q字段 ----------------
iq_field = ''; cand = {'train','test','IQ','iq','iq_data','data','signal','x'};
for i=1:numel(cand), if isfield(S,cand{i}), iq_field = cand{i}; break; end, end
if isempty(iq_field)
    for i=1:numel(fields)
        if isnumeric(S.(fields{i})), iq_field=fields{i}; break; end
    end
end
if isempty(iq_field), error('未识别 I/Q 字段'); end
A = S.(iq_field);
X = unify_to_complex_2D(A); [N,L] = size(X);

% ---------------- 标签（可选） ----------------
label = []; 
for nm = {'trainlabel','testlabel','label','y','labels'}
    if isfield(S, nm{1}), label = S.(nm{1}); label = label(:); break; end
end

% ---------------- 扰动配置 ----------------
PERT = {'CFO','SCALE','GAIN','SHIFT','CHIRP'};

cfg.maxComposite     = 1;          % 先禁多扰动叠加（从3降到1）
cfg.cfo_range_hz     = [-500, 500];% CFO: 2000→500
cfg.scale_range      = [0.98, 1.02];% 时间尺度: 0.95–1.05 → 0.98–1.02
cfg.gain_rho_range   = [0.8, 1.2]; % 幅度: 0.5–1.5 → 0.8–1.2
cfg.shift_samp_ratio = 0.02;       % 移位比例: 0.05 → 0.02
cfg.chirp_a_range    = [-1e3, 1e3];% CHIRP: 5e3 → 1e3

% ---------------- 噪声配置（新增） ----------------
% 方式一：固定 SNR（dB） -> 设 snr_db_fix = 20; cfg.snr_db_range = [];
% 方式二：区间随机 SNR（dB） -> 设 cfg.snr_db_range = [5, 25]; snr_db_fix = [];
snr_db_fix        = [15];            % 例如 20；为空时启用区间随机
cfg.snr_db_range  = [];       % 例如 [8,28] dB；若 snr_db_fix 非空则忽略

% ---------------- 预分配 ----------------
newX = complex(zeros(N,L));
Z    = false(N,numel(PERT)); 
Smat = single(NaN(N,numel(PERT)));
disturb_type   = cell(N,1);
disturb_params = cell(N,1);
snr_db_list    = zeros(N,1,'single');   % 新增：每样本最终SNR(dB)
noise_var_list = zeros(N,1,'single');   % 新增：复噪声方差（每个复样本的总方差）

rng(42);

for i=1:N
    x  = X(i,:); 
    x2 = x;

    % ---- 选扰动 ----
    k  = randi([1, min(cfg.maxComposite, numel(PERT))]);
    chosen = PERT(randperm(numel(PERT), k));
    types={}; params=struct();

    % ---- 逐个扰动叠加 ----
    for t=1:numel(chosen)
        switch chosen{t}
            case 'CFO'
                f0 = urand(cfg.cfo_range_hz);
                x2 = apply_cfo(x2, fs, f0);
                Smat(i,idx('CFO'))   = single(f0); 
                Z(i,idx('CFO'))      = true; 
                params.CFO_f0        = f0; 
                types{end+1}         = 'CFO';

            case 'SCALE'
                a  = urand(cfg.scale_range);
                x2 = apply_scale_timescale(x2, a);
                Smat(i,idx('SCALE')) = single(a); 
                Z(i,idx('SCALE'))    = true; 
                params.SCALE_a       = a; 
                types{end+1}         = 'SCALE';

            case 'GAIN'
                rho = urand(cfg.gain_rho_range); 
                th  = 0; % 如需相位也扰动，可改为 th = urand([-pi,pi]);
                x2  = apply_gain_complex(x2, rho, th);
                Smat(i,idx('GAIN'))  = single(rho); 
                Z(i,idx('GAIN'))     = true; 
                params.GAIN_rho      = rho; 
                params.GAIN_theta    = th; 
                types{end+1}         = 'GAIN';

            case 'SHIFT'
                dmax = round(cfg.shift_samp_ratio * L);
                d    = randi([0, max(dmax,0)]);
                x2   = apply_shift_delay(x2, d);
                Smat(i,idx('SHIFT')) = single(d); 
                Z(i,idx('SHIFT'))    = true; 
                params.SHIFT_delay   = d; 
                types{end+1}         = 'SHIFT';

            case 'CHIRP'
                a   = urand(cfg.chirp_a_range);
                x2  = apply_chirp_phase(x2, fs, a);
                Smat(i,idx('CHIRP')) = single(a); 
                Z(i,idx('CHIRP'))    = true; 
                params.CHIRP_a       = a; 
                types{end+1}         = 'CHIRP';
        end
    end

    % ---- 保持与原信号相同的均方功率（SNR指相对原信号或归一后信号相同）----
    P0 = mean(abs(x).^2); 
    P1 = mean(abs(x2).^2); 
    if P1>0, x2 = x2 * sqrt(P0/P1); end
    Psig = mean(abs(x2).^2);  % == P0

    % ---- 选定目标 SNR(dB) 并加噪（新增） ----
    if ~isempty(snr_db_fix)
        snr_db = snr_db_fix;
    else
        snr_db = urand(cfg.snr_db_range);
    end
    sigma2 = Psig / (10^(snr_db/10));   % 复噪声总方差
    x_noisy = add_cawgn(x2, sigma2);    % 复高斯白噪声，Re/Im 方差各为 sigma2/2

    % ---- 写入结果与记录 ----
    newX(i,:)        = x_noisy;
    disturb_type{i}  = types; 
    disturb_params{i}= params;
    snr_db_list(i)   = single(snr_db);
    noise_var_list(i)= single(sigma2);
end

Z = single(Z);  % [N,5]
assert(all(size(Z)==[N,5]));
assert(all(size(Smat)==[N,5]));
perturb_order = PERT; %#ok<NASGU>

% ---------------- 组装输出 ----------------
Sout = struct();
Sout.(iq_field) = reconstruct_like(A, newX);   % 形状/排布与原数据一致
if ~isempty(label), Sout.trainlabel = label; end
Sout.Z = Z; 
Sout.S = Smat; 
Sout.perturb_order = PERT;
Sout.disturb_type = disturb_type; 
Sout.disturb_params = disturb_params;
Sout.fs = fs; 
Sout.meta = struct('note','5扰动，功率匹配+按SNR加噪','order','[CFO,SCALE,GAIN,SHIFT,CHIRP]');

% 新增：噪声元数据（不改变原有结构）
Sout.snr_db   = snr_db_list;   % [N,1] 每样本目标SNR(dB)
Sout.noise_var= noise_var_list;% [N,1] 每样本复噪声方差（Re/Im各为其一半）

save(outfile, '-struct', 'Sout', '-v7.3');
fprintf('Saved: %s\n', outfile);

    function id=idx(name), id=find(strcmp(PERT,name),1); end
end

% ---------- 工具函数 ----------
function v = urand(r), v = r(1)+(r(2)-r(1))*rand(); end

function X = unify_to_complex_2D(A)
if ~isreal(A)&&isfloat(A), X=A;
elseif ndims(A)==3&&size(A,2)==2, X=complex(squeeze(A(:,1,:)),squeeze(A(:,2,:)));
elseif ndims(A)==3&&size(A,3)==2, X=complex(squeeze(A(:,:,1)),squeeze(A(:,:,2)));
elseif ismatrix(A)&&size(A,1)==2, X=complex(A(1,:),A(2,:));
elseif ismatrix(A)&&size(A,2)==2, X=complex(A(:,1),A(:,2));
else, error('未识别 I/Q 格式');
end
if isvector(X), X=X(:).'; end
end

function out = reconstruct_like(orig, cx)
if ~isreal(orig)&&isfloat(orig), out=cx; return; end
sz=size(orig);
if ndims(orig)==3
    if size(orig,2)==2, out=zeros(sz,'like',orig); out(:,1,:)=real(cx); out(:,2,:)=imag(cx); return; end
    if size(orig,3)==2, out=zeros(sz,'like',orig); out(:,:,1)=real(cx); out(:,:,2)=imag(cx); return; end
elseif ismatrix(orig)
    if size(orig,1)==2, out=zeros(sz,'like',orig); out(1,:)=real(cx); out(2,:)=imag(cx); return; end
    if size(orig,2)==2, out=zeros(sz,'like',orig); out(:,1)=real(cx(:)); out(:,2)=imag(cx(:)); return; end
end
error('无法还原 I/Q。')
end

function y=apply_cfo(x,fs,f0)
n=0:(numel(x)-1); 
y=x.*exp(1j*2*pi*f0*n/fs); 
end

function y=apply_scale_timescale(x,a)
L=numel(x); t=1:L; ts=a*t;
xr=interp1(t,real(x),ts,'linear',0); 
xi=interp1(t,imag(x),ts,'linear',0);
y=complex(xr,xi);
end

function y=apply_gain_complex(x,rho,theta)
y=rho*x*exp(1j*theta); 
end

function y=apply_shift_delay(x,d)
L=numel(x); d=min(max(round(d),0),L-1); 
y=[zeros(1,d),x(1:end-d)]; 
end

function y=apply_chirp_phase(x,fs,a)
n=0:(numel(x)-1); t=n/fs; 
phi=pi*a*(t.^2); 
y=x.*exp(1j*phi); 
end

function y = add_cawgn(x, sigma2)
% 复高斯白噪声：Re/Im ~ N(0, sigma2/2)
noise = sqrt(sigma2/2) * (randn(size(x)) + 1j*randn(size(x)));
y = x + noise;
end
