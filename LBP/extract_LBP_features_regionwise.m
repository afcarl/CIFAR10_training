% Function to compute LMP histogram of an image
% Author: Yogesh Balaji

function LBP_hist = extract_LBP_features_regionwise(img)

	% Convert image to grayscale
	img = rgb2gray(img);
	map = uniform_patterns();
	% Check size 
	sz = size(img);
	if(sz(1)~=32 || sz(2)~=32)
		img = imresize(img, [32 32]);
	end
	assert(size(img,1) == 32 && size(img,2) == 32);
	
	% Computing LBP features for non-overlapping windows
	window_size = 15;
    LBP_hist = [];
	for i=2:window_size:31
		for j=2:window_size:31
            
			tmp = compute_LBP(img(i-1:i+window_size, j-1:j+window_size));
			code = tmp;            
            % Computing LBP histogram
            LBP_hist_local = zeros(59,1);
            for m=1:length(code)
                LBP_hist_local(map(code(m)+1)) = LBP_hist_local(map(code(m)+1))+1;
            end
            LBP_hist_local = LBP_hist_local/norm(LBP_hist_local);
            LBP_hist = [LBP_hist;LBP_hist_local];
		end
	end
	
	
end		
