%% Code for implementing uniform patterns
function map = uniform_patterns()
    code = zeros(8,1);
    code_mul = 2.^(0:7);
    map = ones(256,1);
    counter = 2;
    for num_ones = 0:8
        code = code*0;
        if num_ones == 0
            val = code_mul*code;
            map(val+1) = counter;
            counter = counter+1;
        elseif num_ones == 8
            code = code+1;
            val = code_mul*code;
            map(val+1) = counter;
            counter = counter+1;
        else
            for start = 1:8
                code = code*0;
                for j=start:start+num_ones-1
                    pos = j;
                    if pos>8
                        pos = pos-8;
                    end
                    code(pos) = 1;
                end;
                val = code_mul*code;
                map(val+1) = counter;
                counter = counter+1;
            end

        end
    end
end