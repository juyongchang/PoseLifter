close all
clear
clc

% 1: Site -> 11
% 2: Neck -> 9
% 3: RightArm -> 15
% 4: RightForeArm -> 16
% 5: RightHand -> 17
% 6: LeftArm -> 12
% 7: LeftForeArm -> 13
% 8: LeftHand -> 14
% 9: RightUpLeg -> 2
% 10: RightLeg -> 3
% 11: RightFoot -> 4
% 12: LeftUpLeg -> 5
% 13: LeftLeg -> 6
% 14: LeftFoot -> 7
% 15: Hip -> 1
% 16: Spine -> 8
% 17: Head -> 10

% Mapping table from H36M to MPI
map = [11, 9, 15, 16, 17, 12, 13, 14, 2, 3, 4, 5, 6, 7, 1, 8, 10];

test_subject_id = [1,2,3,4,5,6];
test_data_path = '/media/juyongchang/2e6123d0-bc51-4f5a-a01b-818048d27c37/MPI_INF_3DHP/test/';  %Change to wherever you put this data.
data_base_path = [test_data_path filesep 'TS'];

[~,o1,o2,relevant_labels] = mpii_get_joints('relevant');  

net_base = 'result';

% Load results
load('result.mat');
pred = pred(:,map,:);
pred = pred - pred(:,15,:);
count = 1;

%
sequencewise_per_joint_error = cell(6,1);
sequencewise_activity_labels = cell(6,1);
for i = 1:length(test_subject_id)
   dat = load([data_base_path int2str(test_subject_id(i)) filesep 'annot_data.mat']);
   num_test_points = sum(dat.valid_frame(:));
   per_joint_error = zeros(17,1,num_test_points);
   pje_idx = 1;
   sequencewise_activity_labels{i} = dat.activity_annotation(dat.valid_frame == 1);

    for j = 1:length(dat.valid_frame)
        if(dat.valid_frame(j))
            fprintf('Image %d of %d for Test ID %d\n',j, length(dat.valid_frame), test_subject_id(i));
            error = zeros(17,1);
			
            %img = imread([data_base_path int2str(test_subject_id(i)) filesep 'imageSequence' filesep sprintf('img_%06d.jpg',j)]);
			%The GT has 17 joints, and the order and the annotation of the joints can be observed through the 'relevant_labels' variable
            P = dat.univ_annot3(:,:,:,j)-repmat(dat.univ_annot3(:,15,:,j),1,17);

            %<predict something here>
			pred_p = squeeze(pred(count,:,:))'; %Replace with the actual prediction formatted as 3x17; 
            count = count + 1;
            error_p = (pred_p - P).^2;
            error_p = sqrt(sum(error_p, 1));
            error(:,1) = error(:,1) + error_p(:);


            per_joint_error(:,:,pje_idx) = error;
            pje_idx = pje_idx +1;
        end
    end
    sequencewise_per_joint_error{i} = per_joint_error;
    
end

save([net_base filesep 'mpii_3dhp_prediction.mat'], 'sequencewise_per_joint_error', 'sequencewise_activity_labels');
[seq_table, activity_table] = mpii_evaluate_errors(sequencewise_per_joint_error, sequencewise_activity_labels);

out_file = [net_base filesep 'mpii_3dhp_evaluation'];
writetable(cell2table(seq_table), [out_file '_sequencewise.csv']);
writetable(cell2table(activity_table), [out_file '_activitywise.csv']);

