import numpy as np
from numpy.fft import fft
from scipy.signal import hilbert,find_peaks,firwin
from scipy.signal.windows import hann
from acoustics.cepstrum import real_cepstrum
def ppt(x, fs, fmin=60.0, fmax=400.0):
    """ppt is the short for PWVD Pitch Tracker. For best performance, PLEASE stick to DEFAULT settings.

    Args:
        x    (1d array)  : an input speech signal.
        fs   (float)     : the sampling frequency of x. (note: fs >= 16kHz)
        fmin (float)     : the lowest frequency boundary for pitch tracking, default: fmin=60.0Hz.
        fmax (float)     : the highest frequency boundary for pitch tracking, default: fmax=400.0Hz. 
    Output:
        f0   (1d array)  : the f0 tracking result of x.
        t    (1d array)  : the corresponding sampling time points of f0
    """
    # downsampling
    x = x.flatten()
    fn = np.ceil(2*fmax) # min fs to meet nyquist theorem
    M  = np.floor(fs/fn).astype(int) # downsampling factor
    fd = fs/M # the sampling frequency after downsampling
    xl = LPF(x, fs, fc=fmax)
    xd = xl[::M] # sampled at fd
    # high pass filtering
    xx = HPF(xd, fd, fc=fmin) # sampled at fd, xx's spectrum spans fmin to fmax
    # V/UV decision
    v_idx = V_UV_decision(xx, fd)
    n_voiced = len(v_idx)
    # prepare for real cepstrum
    x = LPF(x, fs, fc=3*fmax)
    # segment pitch tracking
    f0 = np.zeros(len(xx))
    t  = np.arange(len(xx))/fd
    for i in range(n_voiced):
        L = v_idx[i][1]-v_idx[i][0]+1
        f0_seg = seg_pitch_tracking(xx, fd, fmin, fmax, v_idx[i][0], v_idx[i][1], x, fs)
        f0[v_idx[i][0]:v_idx[i][1]+1] = f0_seg[0:L]
    # f0 = populate(f0,MM)
    return f0,t
    
def LPF(x, fs, fc, TBW=20):
    """Low pass filter a signal x.

    Args:
        x   (1d array)       : input signal to be LPFed
        fs  (float)          : the sampling frequency [Hz] of x
        fc  (float)          : the cut-off frequency [Hz]
        TBW (int, optional)  : time bandwidth product. Higher TBW means narrower transition. Defaults to 20.

    Returns:
        xl  (1d array)       : the low pass filtered signal
    """
    Ntaps = (int((fs/2)/fc*2*TBW)//2)*2+1
    h = firwin(Ntaps,fc/(fs/2),window = 'hanning')
    xl = np.convolve(x,h,'same')
    return xl

def HPF(x, fs, fc, TBW=20):
    """High pass filter a signal x.

    Args:
        x   (1d array)       : input signal to be HPFed
        fs  (float)          : the sampling frequency [Hz] of x
        fc  (float)          : the cut-off frequency [Hz]
        TBW (int, optional)  : time bandwidth product. Higher TBW means narrower transition. Defaults to 20.

    Returns:
        xh  (1d array)       : the high pass filtered signal
    """
    Ntaps = (int((fs/2)/fc*2*TBW)//2)*2+1
    h = firwin(Ntaps,fc/(fs/2),window = 'hanning',pass_zero=False)
    xh = np.convolve(x,h,'same')
    return xh

def BPF(x, fs, fc1, fc2, TBW=20):
    """Band pass filter a signal x.

    Args:
        x   (1d array)       : input signal to be HPFed
        fs  (float)          : the sampling frequency [Hz] of x
        fc1 (float)          : the lower cut-off frequency [Hz]
        fc2 (float)          : the higher cut-off frequency [Hz]
        TBW (int, optional)  : time bandwidth product. Higher TBW means narrower transition. Defaults to 20.

    Returns:
        xb  (1d array)       : the band pass filtered signal, spectrum spans fc1 to fc2
    """
    Ntaps = (int((fs/2)/fc1*2*TBW)//2)*2+1
    h = firwin(Ntaps,[fc1/(fs/2), fc2/(fs/2)],window = 'hanning',pass_zero=False)
    xb = np.convolve(x,h,'same')
    return xb

def V_UV_decision(xx, fd):
    """voiced or unvoiced decision based on energy thresholding.

    Args:
        xx (1d array)       : the filtered signal (fmin to fmax)
        fd (float)          : the sampling frequency of xx
    
    Returns: 
        idx (len-m list)    : m is the number of voiced segments, and each element is a 2-element list 
                              containing the start and end indices of voiced segments of xx
    """
    xx = xx.flatten()
    idx = []
    threshold = np.mean(xx**2)
    # print('threshold:',threshold)
    N = len(xx)
    fr_len = np.ceil(25e-3*fd).astype(int) # framesize = 25ms, i.e. 20pts with fd = 800Hz
    n_frame = np.ceil(N/fr_len).astype(int)
    for i in range(1,n_frame+1):
        start = (i-1)*fr_len
        if i != n_frame:
            end_idx = i*fr_len-1
        else:
            end_idx = len(xx)-1
        frame = xx[start:end_idx+1]
        e = np.mean(frame**2)
        # print('frame average energy:',e)
        m = len(frame)
        start1 = start
        end1 = start+np.floor(m/2).astype(int)
        start2 = end1+1
        end2 = end_idx
        sub1 = xx[start1:end1+1]
        sub2 = xx[start2:end2+1]
        e1 = np.mean(sub1**2)
        e2 = np.mean(sub2**2)
        if e > 0.2*threshold:
            if e1 > 0.3*threshold:
                idx = idx_append(start1, end1, idx, xx, fd, threshold)
            if e2 > 0.3*threshold:
                idx = idx_append(start2, end2, idx, xx, fd, threshold)
    # post processing, discarding short segments (less than 25ms)
    n_voiced = len(idx)
    voiced_idx = []
    for j in range(n_voiced):
        if idx[j][1]-idx[j][0] > int(0.025*fd):
            voiced_idx.append([idx[j][0],idx[j][1]])
    return voiced_idx

def idx_append(start, end_idx, idx, x, fd, threshold):
    if idx != [] and start-idx[-1][1] < int(0.060*fd):  # the new frame is next to the previous voiced frame (less than 60ms)
        if start-idx[-1][1] < int(0.032*fd) or np.mean(x[idx[-1][1]:start+1]**2) > 0.05*threshold: # the new frame is very close (less than 32ms)
            idx[-1][1] = end_idx                                                                   # or the gap contains higher energy
    else: 
        idx.append([start, end_idx])
    return idx
    
def seg_pitch_tracking(xx, fd, fmin, fmax, v1, v2, x, fs):
    """voiced segment pitch tracking

    Args:
        xx   (1d array)   : the filtered signal (fmin to fmax)
        fd   (float)      : the sampling frequency of xx
        fmin (float)      : the min analyzed frequency 
        fmax (float)      : the max analyzed frequency 
        v1   (int)        : the starting index of a voiced segment of xx
        v2   (int)        : the end index of a voiced segment of xx
        x    (1d array)   : the original signal after LPFed at fc = 3*fmax, used for cepstrum-based pre-filtering
        fs   (float)      : the sampling frequency of x
    
    Returns:
        f0 (1d array)   : the F0 tracking result of the voiced segment of xx
    """
    # extract the segment interested
    # Note: we accept the assumption that the fundamental frequency is less than 320 Hz, and greater than 70Hz!
    xx  = xx.flatten()    
    seg = xx[v1:v2+1]
    MM = int(fs/fd)
    N = len(seg)
    
    # truncate, supplement, pitch tracking and concatenate
    duration = int(0.150*fd) # each truncated piece should be around 150ms
    n = np.ceil(N/duration).astype(int) # number of truncated pieces
    piece_len = np.floor(N/n).astype(int) # true length of each truncated piece, ~150ms
    # fprintf('piece_len=%f\n',piece_len)
    f0 = np.zeros(N)
    for i in range(1,n+1):
        start = (i-1)*piece_len
        if i != n:
            end_idx = i*piece_len-1
        else:
            end_idx = N-1   
        piece = seg[start:end_idx+1]
        if n == 1:
            add_len_head = min(int(0.013*fd),v1)
            if v2+int(0.013*fd) <= len(xx):
                add_len_tail = int(0.013*fd) 
            else:
                add_len_tail = len(xx)-v2-1
            piece_added = np.hstack([xx[v1-add_len_head:v1],piece,xx[v2+1:v2+add_len_tail+1]]) # supplement head and tail with xx
        elif i == 1:
            add_len_head = min(int(0.013*fd),v1-1)
            add_len_tail = int(0.050*fd)
            piece_added = np.hstack([xx[v1-add_len_head:v1],piece,seg[end_idx+1:end_idx+add_len_tail+1]]) # supplement head (xx) and tail (later info)
        elif i == n:
            add_len_head = int(0.050*fd)
            if v2+int(0.013*fd) <= len(xx):
                add_len_tail = int(0.013*fd) 
            else:
                add_len_tail = len(xx)-v2
            piece_added = np.hstack([seg[start-add_len_head:start],piece,xx[v2+1:v2+add_len_tail+1]]) # supplement head (previous info) and tail (xx)
        else:
            add_len_head = int(0.050*fd)
            add_len_tail = int(0.050*fd)
            piece_added = np.hstack([seg[start-add_len_head:start],piece,seg[end_idx+1:end_idx+add_len_tail+1]]) # supplement head and tail with neighbor's info
        if len(piece) >= int(0.075*fd):
            mid = np.ceil(len(piece)/2).astype(int)
            half_1 = x[(v1+start)*MM:(v1+start+mid)*MM]
            half_2 = x[(v1+start+mid)*MM:(v1+end_idx)*MM+1]
            f0_coarse_1 = coarse_f0_rceps(half_1, fs)
            f0_coarse_2 = coarse_f0_rceps(half_2, fs)
            minor = min(f0_coarse_1,f0_coarse_2)
            delta = abs(f0_coarse_1-f0_coarse_2)
        # if length(piece) >= 60 && abs(f0_coarse_1-f0_coarse_2) > 50 && min(f0_coarse_1,f0_coarse_2) > 100
        if len(piece) >= int(0.075*fd) and ((minor == -1) or (delta/minor > 0.5 and delta/minor < 0.7)):
            # fprintf('(%d, 1st, %d), %f, in x from %d to %d\n', i, n, f0_coarse_1, (v1+start-2)*MM+1, (v1+start+mid-2)*MM+1);
            # fprintf('(%d, 2nd, %d), %f, in x from %d to %d\n', i, n, f0_coarse_2, (v1+start+mid-2)*MM+1,(v1+end_idx-2)*MM+1);
            half_added_1 = piece_added[0:add_len_head+mid+add_len_tail]
            half_added_2 = piece_added[mid+1:]
            add_zero_len_1 = int(0.320*fd)-len(half_added_1)
            add_zero_len_2 = int(0.320*fd)-len(half_added_2)
            f0_raw_1 = piece_pitch_tracking(np.hstack([half_added_1,np.zeros(add_zero_len_1)]),fd,f0_coarse_1,add_zero_len_1,fmin,fmax)
            f0_raw_2 = piece_pitch_tracking(np.hstack([half_added_2,np.zeros(add_zero_len_2)]),fd,f0_coarse_2,add_zero_len_2,fmin,fmax)
            # f0[start:end_idx+1] = np.hstack([f0_raw_1[add_len_head:-add_len_tail-add_zero_len_1],f0_raw_2[add_len_head:-add_len_tail-add_zero_len_2]])
            f0[start:end_idx+1] = np.hstack([f0_raw_1[add_len_head:add_len_head+mid],f0_raw_2[add_len_head:add_len_head+len(piece)-mid]])
        else:
            ori_piece = x[(v1+start)*MM:(v1+end_idx)*MM+1]
            f0_coarse = coarse_f0_rceps(ori_piece, fs)
            # fprintf('(%d, %d), %f, in x from %d to %d\n', i, n, f0_coarse, (v1+start-2)*MM+1, (v1+end_idx-2)*MM+1);
            add_zero_len = int(0.320*fd)-len(piece_added)
            f0_raw = piece_pitch_tracking(np.hstack([piece_added,np.zeros(add_zero_len)]),fd,f0_coarse,add_zero_len,fmin,fmax)
            # f0[start:end_idx] = f0_raw[add_len_head+1:end-add_len_tail-add_zero_len]
            f0[start:end_idx+1] = f0_raw[add_len_head:add_len_head+len(piece)]
    return f0
 
def coarse_f0_rceps(x, fs):
    """returns the coarse F0 of a segment x based on spectrum and real cepstrum. 

    Args:
        x         (1d array)   : a segment of speech signal
        fs        (float)      : the sampling frequency of x
        
    Returns:
        f0_coarse (float)      : the coarse F0 of x 
    """
    # windowing, zero-padding, fft, cepstrum
    add_zero_len = 2**(np.ceil(np.log2(len(x))).astype(int)+2) - len(x)
    x = np.hstack([x*hann(len(x)),np.zeros(add_zero_len)])
    N = len(x)
    delta_f = fs/N
    start_70Hz = np.ceil(70/delta_f).astype(int)
    end_300Hz  = np.floor(300/delta_f).astype(int)
    abs_fft = abs(fft(hilbert(x)))
    idx, info = find_peaks(abs_fft[start_70Hz:end_300Hz+1], distance=start_70Hz, height=0)
    pkv_raw = info['peak_heights']
    if len(pkv_raw) == 0:
        M_fft = np.max(abs_fft[start_70Hz:end_300Hz+1])
    else:
        M_fft = np.max(pkv_raw)

    # gamma candidate selection
    rcps = real_cepstrum(x)
    gamma_290 = np.ceil(fs/290).astype(int) # gamma = fs/f0
    gamma_80  = np.floor(fs/80).astype(int) # gamma = fs/f0
    seg = rcps[gamma_290:gamma_80+1] 
    I = np.argmax(seg)
    gamma_raw = gamma_290+I
    times = np.floor(gamma_80/gamma_raw).astype(int)
    if times > 1:  # gamma_raw: gamma_290~0.5*gamma_80 --> 160Hz~290Hz
        flag = 0
        for j in range(times, 0, -1):
            start = j*gamma_raw-10*int(fs/16000)
            end_idx = j*gamma_raw+10*int(fs/16000)
            seg = rcps[start:min(gamma_80,end_idx)+1]
            i = np.argmax(seg)
            gamma_candidate = start+i
            if choose_candidate(abs_fft, fs, M_fft, gamma_candidate):
                gamma = gamma_candidate
                flag = 1
                break
    else:  # gamma_raw: 0.5*gamma_80 ~ gamma_80 --> 80Hz~160Hz
        flag = 0
        times_2 = np.floor(gamma_raw/gamma_290).astype(int)
        for j in range(1,times_2+1):
            gamma_candidate = np.ceil(gamma_raw/j).astype(int)
            if choose_candidate(abs_fft, fs, M_fft, gamma_candidate):
                gamma = gamma_candidate
                flag = 1
                break
    if flag == 0:
        f0_coarse = -1
        print('gamma_raw is false!')
    else:
        f0_coarse = fs/gamma
    return f0_coarse

def choose_candidate(abs_fft, fs, M_fft, gamma_candidate):
    """check if there is a clear spectrum peak in the frequency range indicated by gamma_candidate.
    
    Args:
        abs_fft         (1d array) : the absolute values of the fft of the analyzed segment
        fs              (float)    : the sampling frequency of the analyzed segment
        M_fft           (float)    : the max peak value of abs_fft in the interested frequency range
        gamma_candidate (int)      : a gamma candidate given by cepstrum peak
        
    Returns:
        sign            (bool)     : True if a clear spectrum peak is detected; False otherwise
    """
    N = len(abs_fft)
    delta_f = fs/N
    pos = np.ceil(N/gamma_candidate).astype(int)
    range_30Hz = np.ceil(30/delta_f).astype(int)
    pos_70Hz = np.floor(70/delta_f).astype(int)
    pos_300Hz = np.ceil(300/delta_f).astype(int)
    seg = abs_fft[max(pos-range_30Hz,pos_70Hz):min(pos+range_30Hz,pos_300Hz)+1]
    candidate_idx, _ = find_peaks(seg,height=0.1*M_fft)
    if len(candidate_idx) == 0:
        sign = False
    else:
        sign = True
    return sign

def piece_pitch_tracking(xx,fd,f0_coarse,add_zero_len,fmin,fmax):
    """pitch tracking for a short voiced piece

    Args:
        xx              (1d array) : a piece of a voiced segment 
        fd              (float)    : the sampling frequency of xx
        f0_coarse       (float)    : the coarse f0 of xx
        add_zero_len    (int)      : length of added zeros
        fmin            (float)    : the min analyzed frequency
        fmax            (float)    : the max analyzed frequency
    
    Returns:
        f0              (1d array) : an array of fundamental frequencies at each sample of xx
    """
    xx = xx.flatten()
    N  = len(xx)
    f0 = np.zeros(N)
    # check if f0_coarse is valid
    if f0_coarse != -1:
        # pre-filter to keep 0.7*f0_coarse to 1.4*f0_coarse frequency components
        xx = BPF(xx, fd, fc1=max(0.7*f0_coarse,fmin), fc2=min(1.4*f0_coarse,fmax))
        # take the analytic signal of x to avoid interpolation and cross terms
        xz      = hilbert(xx)
        xa      = np.hstack([xz,np.zeros(N)])
        xa_conj = np.hstack([np.zeros(N),np.conj(xz)])
        delta_f = fd/2/(N+1)
        start_fmin = np.floor(fmin/(fd/2)*(N+1)).astype(int)
        # PWVD
        for i in range(N-add_zero_len):
            w = hann(min(int(0.038*fd),N+1))
            w = w[np.ceil((len(w)+1)/2).astype(int):]
            w = np.hstack([w,np.zeros(N+1-len(w))])
            R = np.flip(xa_conj[i:i+N+1])*xa[i:i+N+1]*(w**2)
            fft_R = fft(R)
            d = np.real(fft_R + np.conj(fft_R) - xa[i]*np.conj(xa[i])*w[0]**2)
            I = detect_1st_peak(d[start_fmin:], fd)
            if I != -1:
                f0[i] = (I+start_fmin)*delta_f
    return f0

def detect_1st_peak(d, fd):
    """select the first prominent peak from PWVD

    Args:
        d   (1d array) : the PWVD at a certain time point 
        fd  (float)    : the sampling frequency at which PWVD is calculated
    
    Returns:
        idx (int)      : the index of the detected first prominent peak 
    """
    N = len(d)
    # we accept the assumption that f0 > 60Hz, therefore we set MinPeakDistance to be 30Hz (considering cross terms)
    locs, info = find_peaks(d, distance=np.ceil(30/(fd/2)*N).astype(int), height=0.1*max(d))
    pkv = info['peak_heights']
    threshold = np.mean(pkv)*0.5
    if len(pkv) == 0:
        idx = -1
    else:
        i = 0
        while pkv[i] < threshold:
            i += 1
        idx = locs[i]
    return idx


    

