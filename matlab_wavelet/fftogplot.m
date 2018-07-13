function [FTsignal] = fftogplot(h,N,fs)
% Funksjon som foretar en fft av et signal h med N punkter.
% Samplingsfrekvensen er fs. Den fouriertransformerte
% plottes (absuluttverdien), mens det komplekse
% fouriertransformerte signalet returneres til den kallende funksjon.
% Versjon 20.4.2016



y = load('B.mat');
h = y.B;
%N = y.N;
%fs = y.fs;
fs = 2856.56921073 ;
N = 171394 %1704634;

% Beregner først FFT av tidsstrengen h
FTsignal = fft(h);
% Plotter frekvensspekteret (absoluttverdier only)
f = linspace(0,fs*(N-1)/N, N);
nmax = floor(N/2);    % Plotter bare opp til halve samplingsfrekv.
figure;
plot(f(1:nmax),abs(FTsignal(1:nmax)));
xlabel('Frekvens (Hz)');
ylabel('Relativ intensitet');
title('Frekvensspektrum av signal');