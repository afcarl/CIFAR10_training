% Function to compute LMP histogram of an image
% Author: Yogesh Balaji

function LBP_hist = extract_LBP_features(img)

	% Convert image to grayscale
	img = rgb2gray(img);
	
	% Check size 
	sz = size(img);
	if(sz(1)~=32 || sz(2)~=32)
		img = imresize(img, [32 32]);
	end
	assert(size(img,1) == 32 && size(img,2) == 32);
	
	% Computing LBP features for non-overlapping windows
	window_size = 30;
	code = [];
	for i=2:window_size:31
		for j=2:window_size:31
			tmp = compute_LBP(img(i-1:i+window_size, j-1:j+window_size));
			code = [code; tmp];
		end
	end
	
	% Computing LBP histogram
	LBP_hist = zeros(256,1);
	for i=1:length(code)
		LBP_hist(code(i)+1) = LBP_hist(code(i)+1)+1;
	end
	LBP_hist = LBP_hist/norm(LBP_hist);
end		
