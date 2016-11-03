% function for extracting LBP features for a patch - Here we are consider
% To extract the LBP features for a patch of dimensions nXn, pass (n+2)X(n+2) size patch as input. The extra 2 rows and 2 columns are for the top and bottom rows and colums.
 
function encoding_array = compute_LBP_regionwise(patch)
	
	sz = size(patch);
	lspace  = (0:7)';
	code_mul = zeros(8,8);
	for i=0:7
		code_mul(i+1,:) = circshift(lspace,i)';
	end
	code_mul = 2.^code_mul;
	mappings = [1 2 3 5 8 7 6 4]';
	
	encoding_array = zeros((sz(1)-2)*(sz(2)-2),1);
	k = 1;
	for i=2:sz(1)-1
		for j=2:sz(2)-1
			tmp_patch = reshape(patch(i-1:i+1, j-1:j+1),9,1);
			prime_val = tmp_patch(5,1);
			tmp_patch(5) = [];
			tmp_patch2 = tmp_patch(mappings);
			code = tmp_patch2>=prime_val;
			encoding = min(code_mul*code);
			encoding_array(k,1) = encoding;
			k = k+1;
		end
	end
end	

