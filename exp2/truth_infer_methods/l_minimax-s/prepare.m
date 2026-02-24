
addpath(genpath(pwd));

ai_names = ["g100", "f100", "l250", "mix"];
ai_nums = [0,3,6,9];
for ai_name = ai_names
    for ai_num = ai_nums
        for iii = 1:5
            rng(iii*777);
            filename = "__ds_" + ai_name + "_" + ai_num + "__.csv";
            %% prepare answer Matrix
            fid = fopen(filename, 'r', 'n', 'UTF-8'); % question, worker, answer
            fgetl(fid); % drop the first line
            data = textscan(fid, '%s %s %s', 'Delimiter', ',');
            fclose(fid);
            if ~exist('uniqueQ', 'var')
                uniqueQ = unique(data{1});
            end
            
            uniqueW = unique(data{2});
            L = zeros(length(uniqueQ), length(uniqueW));
            for i = 1:length(data{1})
                Qindex = find(strcmp(uniqueQ, data{1}(i)));
                Windex = find(strcmp(uniqueW, data{2}(i)));
                L(Qindex, Windex) = str2double(data{3}(i)) + 1; % plus 1 to make all labels positive, 0 indicates no label 
            end
            
            clearvars -except L uniqueQ iii ai_name ai_num filename ai_nums ai_names;
            
            Model = crowd_model(L);
            
            lambda_worker = 0.25*Model.Ndom^2; lambda_task = lambda_worker * (mean(Model.DegWork)/mean(Model.DegTask)); % regularization parameters
            opts={'lambda_worker', lambda_worker, 'lambda_task', lambda_task, 'maxIter',50,'TOL',5*1e-3','verbose',1};
            % 1. Categorical minimax entropy:
            result1 =  MinimaxEntropy_crowd_model(Model,'algorithm','categorical',opts{:});
            writecsv("result_" + ai_name + "_" + ai_num + "_" + iii + ".csv", uniqueQ, result1.soft_labels);
        end
    end
end