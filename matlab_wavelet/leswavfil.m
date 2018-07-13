function [fs,h] = leswavfil(c,nstart,N)
% Denne funksjonen leser en wav-fil ned filnavn c.
% Vi starter lesingen nstart punkter etter filstarten,
% og det leses N datapunkter fra filen.
% Lyden spilles, og signalet plottes.
% Funksjonen returnerer samplingsfrekvensen og
% den ene kanalen av stereosignalet i wav-filen.
% Denne versjonen er fra 20. april 2016
%nslutt = nstart+N-1;
%[y, fs] = audioread(c, [nstart nslutt]); % Les array y(N,2) fra fil
% ’fs’ er vanligvis 44100 (samplingsfrekvens ved CD kvalitet)


%filename='USt37.dat'; 
%fid = fopen(filename, 'rt');
%data = cell2mat( textscan(fid, '%f%f', 'HeaderLines', 3, 'CollectOutput', 1) );
%fclose(fid)

y = load('B.mat');
h = y.B
N = y.N
fs = y.fs;


%y = load('B.mat');
%fs = 2856.56921073 ;

%N = 1704634;
%h = y.B;

%h = linspace(0, floor(fs*N), N);

% h = zeros(N,1);    % Plukker ut bare én kanal fra stereosignalet lest
% h = y(:,1);
% sound(h,fs);          % Spiller av utsnittet som er brukt
T = N/fs;                % Total tid lydutsnittet tar (i sek)
t = linspace(0,T*(N-1)/N,N);
plot(t,h,'-k');
title('Wav-filens signal');
xlabel('Tid (sek)');
ylabel('Signal (rel enhet)');