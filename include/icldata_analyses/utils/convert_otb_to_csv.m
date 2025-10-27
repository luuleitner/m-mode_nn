% CMD line usage: matlab -batch "convert_otb_to_csv('C:\Users\chule\Downloads\0.otb+')"

function convert_otb_to_csv(inputFilePath)
    % Convert an OTB+ or ZIP file to a CSV file.
    % inputFilePath: Full path to the .otb+ or .zip file

    % Check input
    if nargin < 1 || ~isfile(inputFilePath)
        error('Please provide a valid .otb+ or .zip file path.');
    end

    % Create temporary extraction folder
    tempFolder = 'tmpopen';
    if exist(tempFolder, 'dir')
        rmdir(tempFolder, 's');
    end
    mkdir(tempFolder);

    % Extract archive
    [~, ~, ext] = fileparts(inputFilePath);
    if strcmp(ext, '.zip') || strcmp(ext, '.otb+')
        untar(inputFilePath, tempFolder);
    else
        error('Unsupported file type: %s. Expected .otb+ or .zip', ext);
    end

    % Process signals
    signals = dir(fullfile(tempFolder, '*.sig'));
    if isempty(signals)
        error('No .sig files found in archive.');
    end

    for nSig = 1:length(signals)
        % Parse XML metadata
        xmlFile = fullfile(tempFolder, [signals(nSig).name(1:end-4) '.xml']);
        abs = xml2struct(xmlFile);
        
        sampleFreq = str2double(abs.Device.Attributes.SampleFrequency);
        nChannels = str2double(abs.Device.Attributes.DeviceTotalChannels);
        nADBit = str2double(abs.Device.Attributes.ad_bits);
        powerSupply = 3.3;

        % Initialize gains
        gains = zeros(1, nChannels);
        adapters = abs.Device.Channels.Adapter;

        for nChild = 1:length(adapters)
            adapter = adapters{nChild};
            localGain = str2double(adapter.Attributes.Gain);
            startIdx = str2double(adapter.Attributes.ChannelStartIndex);

            channels = adapter.Channel;
            for nChan = 1:length(channels)
                if iscell(channels)
                    channelAtt = channels{nChan}.Attributes;
                else
                    channelAtt = channels(nChan).Attributes;
                end
                idx = str2double(channelAtt.Index);
                gains(startIdx + idx + 1) = localGain;
            end
        end

        % Load signal data
        h = fopen(fullfile(tempFolder, signals(nSig).name), 'r');
        rawData = fread(h, [nChannels, Inf], 'short');
        fclose(h);

        % Scale signal
        scaledData = zeros(size(rawData));
        for nCh = 1:nChannels
            scaledData(nCh, :) = rawData(nCh, :) * powerSupply / (2^nADBit) * 1000 / gains(nCh);
        end

        % Transpose and export to CSV
        csvFileName = [signals(nSig).name(1:end-4) '.csv'];
        csvwrite(csvFileName, scaledData');

        fprintf('Saved %s\n', csvFileName);
    end

    % Optionally process *.sip / *.pro files if needed (commented out)
    % processProcessed(tempFolder);

    % Clean up
    rmdir(tempFolder, 's');
end
