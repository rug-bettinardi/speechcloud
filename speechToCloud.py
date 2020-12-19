import os
import shutil
import logging

from pydub import AudioSegment, effects
import speech_recognition as sr
from stop_words import get_stop_words
from wordcloud import WordCloud
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def extractAndSaveWavSegment(srcFile, tgtWavFile, minutes):
    """
    Extract & save only the desired segment from an audio file, in WAV format.

    Args:
        srcFile: (str) full path of source audio file
        tgtWavFile: (str) full path of target WAV audio file
        minutes: (list of int) with [startMin, stopMin] of the segment of interest

    Returns:
        chunk: pydub.AudioSegment of interest

    """

    _, file = os.path.split(srcFile)
    fileName, fileFormat = file.split(".")
    audiofile = AudioSegment.from_file(srcFile, format=fileFormat)

    oneMinMs = 60 * 1000  # pydub works in milliseconds
    startMs = minutes[0] * oneMinMs
    stopMs = minutes[1] * oneMinMs
    durationMs = len(audiofile)
    durationMin = round((durationMs / 1000) / 60, 1)

    if startMs < 0:
        raise ValueError("startMin cannot be a negative number!")

    if stopMs > durationMs:
        print(f"provided stopMin (={minutes[1]} min) is larger than the original audio file duration (={durationMin} min): "
              f" extracted interval is therefore [{minutes[0]}, {durationMin}] minutes")
        stopMs = durationMs

    chunk = audiofile[startMs:stopMs]
    chunk.export(os.path.join(tgtWavFile), format="wav")

    return chunk


def splitAndSaveAudio(srcFile, tgtDir, chunkDurationMin, tgtFormat="wav", dBChangeVolume=None):
    """
    use pydub to split and save original mp3 file into chunks
    of given duration.

    Args:
        srcFile: (str) full path to audio file
        tgtDir: (str) path to target directory where chunks will be stored
        chunkDurationMin: (int or float) desired chunk duration
        tgtFormat: (str) default is "wav"
        dBChangeVolume: (int) num of dB to scale up (+) or down (-) the audio volume [default = None, automatic]

    Returns:

    """

    _, file = os.path.split(srcFile)
    fileName, fileFormat = file.split(".")
    audiofile = AudioSegment.from_file(srcFile, format=fileFormat)
    durationMs = len(audiofile)
    audiofile = effects.normalize(audiofile)

    if dBChangeVolume:
        # audiofile = audiofile + dBChangeVolume
        audiofile = audiofile.apply_gain(dBChangeVolume)
    else:
        audioVolume = audiofile.dBFS
        volumeLowThr = -20
        if audioVolume < volumeLowThr:
            audiofile = audiofile.apply_gain(abs(audioVolume - volumeLowThr))
            # audiofile = audiofile + abs(audioVolume - volumeLowThr)

    oneMinMs = 60 * 1000  # pydub works in milliseconds
    chunkDurationMs = chunkDurationMin * oneMinMs

    counter = 1
    startMs = 0
    stopMs = startMs + chunkDurationMs
    while startMs <= durationMs:

        if stopMs > durationMs:
            stopMs = durationMs

        chunkFile = f"{fileName}_{counter}.{tgtFormat}"
        chunk = audiofile[startMs:stopMs]
        chunk.export(os.path.join(tgtDir, chunkFile), format=tgtFormat)
        del chunk

        startMs = stopMs + 1
        stopMs = startMs + chunkDurationMs
        counter += 1

    del audiofile
    print(f"splitAndSaveAudio, {file}: DONE")


def speechToText(wavFile, language="it-IT", engine="googleCloudAPI"):
    """

    Args:
        wavFile: (str) full path to wav audio file
        language: (str)
        engine: (str) either "googleCloudAPI" [default] or "google"

    Returns:
        text

    """

    recog = sr.Recognizer()

    with sr.AudioFile(wavFile) as source:
        recog.adjust_for_ambient_noise(source)
        audio = recog.record(source)

    try:

        if engine == "google":
            text = recog.recognize_google(audio, language=language)

        elif engine == "googleCloudAPI":

            with open(r"P:\WORK\PYTHONPATH\rug\docs\radiolockdown-0961f41fa3dc.json") as f:
                credentialsGoogleCloudAPI = f.read()

            text = recog.recognize_google_cloud(audio, language=language, credentials_json=credentialsGoogleCloudAPI)

        else:
            print(f"'{engine}' is not a recognized speech-recognition engine. returning empty text")
            text = ""

        return text

    except Exception as e:
        file = os.path.split(wavFile)[1]
        print(f"speechToText on: {file} not possible, returning ''\n --> {e}")
        emptyString = ""
        return emptyString


def textFromMultipleSpeechFiles(wavDir, language="it-IT", engine="googleCloudAPI"):
    """

    Args:
        wavDir: (str) full path to directory containing wav audio files (each smaller then 10 MB!!!)
        language:
        engine: (str) either "googleCloudAPI" [default] or "google"

    Returns:
        txt: (str)

    """

    longTxt = []

    for wav in os.listdir(wavDir):
        wavFilePath = os.path.join(wavDir, wav)

        try:
            txt = speechToText(wavFilePath, language=language, engine=engine)

        except:
            print(f"{wav}: didn't work ... skipping it")
            txt = ""

        longTxt.append(" " + txt)

    return " ".join(longTxt)


def getStopWords(text, language='it', stopList=None):
    """
    Remove every word that is neither a NOUN or a PROPN

    Args:
        text: (str)
        language: (str) default is 'it'
        stopList: (list of str) of additional stop words

    Returns:
        stopWords: (list of str)

    """

    import spacy

    if language == 'it':
        nlp = spacy.load('it_core_news_lg')
        stopWords = get_stop_words('it') + stopList
    else:
        print(f"{language} still not implemented n this function, returning empty list")
        return []

    doc = nlp(text)

    for token in doc:

        notNounOrProp = (token.pos_ != 'NOUN') and (token.pos_ != 'PROPN')

        if notNounOrProp:
            stopWords.append(token.text)

    return stopWords


def plotWordCloud(text, stopwords=None, max_words=200, background_color="white", colormap="viridis", **kwargs):
    """
    Plot Wordcloud

    Args:
        text: text to convert to wordcloud
        max_words: max num of words to plot
        stopwords: list of str, words to remove from the computation
        background_color: str [default = "white"]
        colormap: matplotlib colormap [default = "viridis"]

    Returns:
        fig handle

    """

    defaultKwargs = {
        "width": 3000,
        "height": 2000,
        "random_state": 1,
        "collocations": True,
    }

    if kwargs:
        for k in defaultKwargs:
            if k not in kwargs.keys():
                kwargs[k] = defaultKwargs[k]
    else:
        kwargs = defaultKwargs

    if stopwords is not None and not isinstance(stopwords, set):
        stopwords = set(stopwords)

    fig = plt.figure(figsize=(19.0, 19.0))

    try:
        wordcloud = WordCloud(stopwords=stopwords,
                              max_words=max_words,
                              background_color=background_color,
                              colormap=colormap,
                              **kwargs).generate(text)

        plt.imshow(wordcloud)
        plt.axis("off")

    except Exception as E:
        print(f"plotWordCloud: {E}")

    return fig


class SpeechCloud:

    def __init__(self, language=None, engine=None, minutes=None, dBChangeVolume=None):
        """

        Args:
            language: (str) default is 'it'
            engine: (str) either "googleCloudAPI" [default] or "google"
            minutes: (list of int) storing the start and end minutes of the audio to transcribe (and plot) [default = None, all audio]
            dBChangeVolume: (int) num of dB to scale up (+) or down (-) the audio volume [default = 0, no change]

        """

        self.language = 'it' if language is None else language
        self.engine = engine if engine is not None else "googleCloudAPI"
        self.minutes = minutes
        self.dBChangeVolume = dBChangeVolume
        self.text = None
        self._tempDir = r"P:\WORK\PYTHONPATH\rug\projects\speechcloud\temp"  # TODO: automatically in "speechcloud" dir

        if not os.path.exists(self._tempDir):
            os.mkdir(self._tempDir)

    def transcribe(self, audio):
        """

        Args:
            audio: filePath, list of filePaths, or path to directory containing all the audio files

        Returns:
            text: (str) speech-to-text transcript (concatenating all audio files if audio is not a single file)

        """

        if isinstance(audio, list):

            transcripts = []
            for filePath in audio:

                if os.path.isfile(filePath):
                    transcripts.append(self._getTextFromOneFile(srcFilePath=filePath))

                else:
                    print(f"{filePath} is not a recognized argument format. returning empty transcript.")
                    transcripts.append("")

            self.text = " ".join(transcripts)

        elif os.path.isdir(audio):

            transcripts = []
            for file in os.listdir(audio):
                filePath = os.path.join(audio, file)

                if os.path.isfile(filePath):
                    transcripts.append(self._getTextFromOneFile(srcFilePath=filePath))

                else:
                    print(f"{filePath} is not a recognized argument format. returning empty transcript.")
                    transcripts.append("")

            self.text = " ".join(transcripts)

        elif os.path.isfile(audio):
            self.text = self._getTextFromOneFile(srcFilePath=audio)

        else:
            raise ValueError(f"{audio} is not a recognized argument format")

        # remove all temporary audio files:
        shutil.rmtree(self._tempDir)

    def plot(self, audio=None, stopwords=None, max_words=200, background_color="white", colormap="viridis", **kwargs):

        """
             mask=None, scale=1,
             figsize=(10, 10), collocation_threshold=30,
             font_path=None, margin=2, ranks_only=None, prefer_horizontal=.9,
             color_func=None, min_font_size=4, max_font_size=None, font_step=1, mode="RGB",
             relative_scaling='auto', regexp=None, normalize_plurals=True, contour_width=0,
             contour_color='black', repeat=False, include_numbers=False, min_word_length=0:

        """

        defaultKwargs = {
            "width": 3000,
            "height": 2000,
            "random_state": 1,
            "collocations": True,
        }

        if kwargs:
            for k in defaultKwargs:
                if k not in kwargs.keys():
                    kwargs[k] = defaultKwargs[k]
        else:
            kwargs = defaultKwargs

        if audio:
            self.transcribe(audio)

        if self.text:
            STOPWORDS = getStopWords(self.text, language=self.language, stopList=stopwords)
            figureHandle = plotWordCloud(text=self.text, stopwords=STOPWORDS, max_words=max_words,
                                         background_color=background_color, colormap=colormap, **kwargs)
            return figureHandle

        else:
            raise ValueError("No transcription already stored. Please provide 'audio' input argument")

    def getTranscription(self):
        warnLog = "No transcription stored. run SpeechCloud.transcribe(audio) first"
        return self.text if self.text else logger.warning(warnLog)

    def saveTranscription(self, tgtFile):
        warnLog = "No transcription stored. run SpeechCloud.transcribe(audio) first"
        if self.text:

            with open(tgtFile, "w+") as txtFile:
                txtFile.write(self.text)

        else:
            logger.warning(warnLog)

    def _getTextFromOneFile(self, srcFilePath):
        """
        Internal method. Perform speech-to-text from one audio file
        using the parameters defined in .__init__ and .transcribe

        Args:
            srcFilePath: (str) full path to audio file

        Returns:
            text: (str) transcription obtained from speech-to-text

        """

        if self.minutes:
            # create temporary child folder:
            segmentFolder = os.path.join(self._tempDir, "segment")
            if not os.path.exists(segmentFolder):
                os.mkdir(segmentFolder)

            # extract only desired audio segment:
            _, file = os.path.split(srcFilePath)
            fileName = file.split(".")[0]
            tgtFilePath = os.path.join(segmentFolder, f"{fileName}_segment.wav")
            _ = extractAndSaveWavSegment(srcFile=srcFilePath, tgtWavFile=tgtFilePath, minutes=self.minutes)
            del _

        else:
            # use all audio file:
            tgtFilePath = srcFilePath

        # create temporary child folder:
        chunksFolder = os.path.join(self._tempDir, "chunks")
        if not os.path.exists(chunksFolder):
            os.mkdir(chunksFolder)

        splitAndSaveAudio(srcFile=tgtFilePath,
                          tgtDir=chunksFolder,
                          chunkDurationMin=1,
                          tgtFormat="wav",
                          dBChangeVolume=self.dBChangeVolume)

        text = textFromMultipleSpeechFiles(chunksFolder, language=self.language, engine=self.engine)

        return text


if __name__ == "__main__":

    audioGerry = [
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\esposizione.m4a",
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta1.m4a",
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta2.m4a",
    ]
    inputLst = [
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\esposizione.m4a",
        r"P:\WORK\PYTHONPATH\rug\projects\autoradiolockdown\ruggero-dev\autolog\audio\puntate\mp3\segmentsMp3\puntata23\noSong_10.mp3",
        r"P:\WORK\PYTHONPATH\rug\projects\autoradiolockdown\ruggero-dev\autolog\audio\puntate\mp3\segmentsMp3\puntata23\noSong_3.mp3",
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta1.m4a",
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta2.m4a",
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio",  # directory
        audioGerry,  # list
    ]

    STOPWORDS = [
        "Luca", "Dani", "Daniele", "Chiara", "Marco", "Leo", "Micol", "Rugge", "Ali", "Pat", "Beppe", "Ste", "Claudio",
        "ragazzi", "ragazze", "raga", "l'altro", "l'altra", "volta", "l'ho", "l'hai", "l'ha", "volte", "l'hanno",
        "dell'altro", "dell'altra", "c'era", "c'erano", "l'ultima", "l'ultimo", "cos'è", "cos'era", "l'abbiamo",
        "cos'erano", "roba", "ni", "realtà", "c'eravate", "c'eravamo", "tanto", "s'è", "glielo", "dall'altro",
        "dall'altra", "cos'altro", "l'è", "cosa", "cose", "tant'è", "meglio", "casi", "po'", "n'era", "gran", "Digli",
        "puntata", "tema", "proposito", "caso", "tipo", "tipa", "tipi", "esempi", "esempio", "Luino", "Ruggiero",
        "anno", "modo", "modi", "Stefano", "Vicente", "Chat", "chat", "com", "posto", "posti", "c'è", "grazie"
    ]

    for audio in inputLst:
        print(audio)

        # path, file = os.path.split(audio)
        # fmt = file.split(".")[1]
        #
        # audiofile = AudioSegment.from_file(audio, format=fmt)
        # print(f"{file}: duration = {round(audiofile.duration_seconds / 60, 1)} minutes, dBFS = {audiofile.dBFS}")

        # sc = SpeechCloud(language='it')
        sc = SpeechCloud(language='it', minutes=[1, 3])
        sc.plot(audio=audio, stopwords=STOPWORDS, max_words=400, colormap="inferno")

        text = sc.getTranscription()
        print(text)

        plt.close()

    print("FINISHED")

    ##

    # sc = SpeechCloud(language='it', dBChangeVolume=25)
    # sc = SpeechCloud(language='it', minutes=[1, 2], dBChangeVolume=25)
    # # sc.plot(stopwords=STOPWORDS, max_words=400, colormap="inferno", random_state=2)
    # sc.plot(audio=AUDIO, stopwords=STOPWORDS, max_words=400, colormap="inferno")


    # src = r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\esposizione.m4a"
    # tgt = r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\esposizione_segment.wav"
    # extractAndSaveWavSegment(srcFile=src, tgtWavFile=tgt, minutes=[1,2])

    sc.plot(audio=audioGerry, stopwords=STOPWORDS, max_words=400, colormap="inferno")