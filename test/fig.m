%% PSNR

% for i = 1:5
% 	psnr = load([num2str(i), '_psnr.txt']);
% 	figure(i);
% 	x = psnr(:,1);
% 	psnr_ref = psnr(:,2);
% 	psnr_my = psnr(:,3);
% 	plot(x, psnr_ref, 'r', x, psnr_my, 'b');
% 	legend('Reference\_PSNR', 'My\_PSNR', 'Location', 'NorthWest');
% 	xlabel('quality');
% 	ylabel('dB');
% end

%% Size

% for i = 1:5
% 	size = load([num2str(i), '_size.txt']);
% 	figure(i);
% 	x = size(:,1);
% 	size_ref = size(:,2);
% 	size_my = size(:,3);
% 	plot(x, size_ref, 'r', x, size_my, 'b');
% 	legend('Reference\_Size', 'My\_Size', 'Location', 'NorthWest');
% 	xlabel('quality');
% 	ylabel('bytes');
% end

%% PSNR4

% figure(1);
% for i = 2:5
% 	psnr = load([num2str(i), '_psnr.txt']);
% 	x = psnr(:,1);
% 	y = psnr(:,3);
% 	plot(x, y, 'lineWidth', 2);
% 	hold on;
% end
% xlabel('quality');
% ylabel('dB');
% legend('2', '3', '4', '5', 'Location', 'NorthWest');

%% Size4

figure(2);
for i = 2:5
	size = load([num2str(i), '_size.txt']);
	x = size(:,1);
	y = size(:,3);
	plot(x, y, 'lineWidth', 2);
	hold on;
end
xlabel('quality');
ylabel('bytes');
legend('2', '3', '4', '5', 'Location', 'NorthWest');