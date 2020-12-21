import os
import shutil
import logging

from pydub import AudioSegment, effects
import speech_recognition as sr
from stop_words import get_stop_words
from wordcloud import WordCloud
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def extractDesiredAudioSegment(srcFile, minutes, tgtWavFile=None):
    """
    Extract only the desired segment from an audio file, and optionally save it in WAV format.

    Args:
        srcFile: (str) full path of source audio file
        minutes: (list of int) with [startMin, stopMin] of the segment of interest
        tgtWavFile: (str) full path of target WAV audio file [default = None, does not save segment]

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

    if tgtWavFile:
        chunk.export(os.path.join(tgtWavFile), format="wav")

    return chunk


def splitIntoShorterAudioChunks(srcFile, tgtDir, chunkDurationMin, tgtFormat="wav", dBChangeVolume=None):
    """
    use pydub to split and save original mp3 file into smaller
    chunks of given duration.

    Args:
        srcFile: (str) full path to audio file
        tgtDir: (str) path to target directory where chunks will be stored
        chunkDurationMin: (int or float) desired chunk duration
        tgtFormat: (str) default is "wav"
        dBChangeVolume: (int) num of dB to scale up (+) or down (-) the audio volume [default = None, automatic]

    Returns:
        None, it only saves the split chunks

    """

    _, file = os.path.split(srcFile)
    fileName, fileFormat = file.split(".")
    audiofile = AudioSegment.from_file(srcFile, format=fileFormat)
    durationMs = len(audiofile)
    audiofile = effects.normalize(audiofile)

    if dBChangeVolume:
        audiofile = audiofile.apply_gain(dBChangeVolume)
    else:
        audioVolume = audiofile.dBFS
        volumeLowThr = -20
        if audioVolume < volumeLowThr:
            audiofile = audiofile.apply_gain(abs(audioVolume - volumeLowThr))

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
    print(f"splitIntoShorterAudioChunks, {file}: DONE")


def speechToText(wavFile, language="it-IT", engine="googleCloudAPI"):
    """
    wrapper of speech_recognition module.

    Args:
        wavFile: (str) full path to wav audio file
        language: (str) default is 'it-IT'
        engine: (str) either "googleCloudAPI" [default] or "google"

    Returns:
        text: (str) obtained from speech-to-text

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
    Perform speech-to-text on multiple audio files and
    concatenate all text outputs.

    Args:
        wavDir: (str) full path to directory containing wav audio files (each smaller then 10 MB!!!)
        language: (str) default is 'it-IT'
        engine: (str) either "googleCloudAPI" [default] or "google"

    Returns:
        txt: (str) concatenated output

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
    Return list of words that are NOT a NOUN nor a PROPN

    Args:
        text: (str) source text to clean
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


def plotWordCloud(text, language="it", stopwords=None, max_words=200,
                  background_color="white", colormap="viridis",
                  figsize=(10.0, 10.0), **kwargs):
    """
    Wrapper of WordCloud [*]. Plot Wordcloud of a given text.

    Args:
        text: text to convert to wordcloud
        language: (str) default is 'it'
        stopwords: list of str, words to remove from the computation
        max_words: max num of words to plot
        background_color: str [default = "white"]
        colormap: matplotlib colormap [default = "viridis"]
        figsize: (float, float) matplotlib figure figsize [default = (10.0, 10.0)]
        kwargs: additional arguments of WordCloud [*]. examples of kwargs are: font_path, color_func, mask, ...
                - To create a word cloud with a single color, use: ``color_func=lambda *args, **kwargs: "white"``
    Returns:
        figure handle

    [*] http://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud

    """

    defaultKwargs = {
        "width": 3000,
        "height": 2000,
        "random_state": 1,
        "collocations": True,
    }

    # update kwargs with default values if not provided:
    if kwargs:
        for k in defaultKwargs:
            if k not in kwargs.keys():
                kwargs[k] = defaultKwargs[k]
    else:
        kwargs = defaultKwargs

    STOPWORDS = set(getStopWords(text, language=language, stopList=stopwords))

    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    try:
        wordcloud = WordCloud(stopwords=STOPWORDS,
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

    def transcribe(self, audio):
        """
        perform speech-to-text of given audio argument (see Args) and update self.text class attribute 
        with the resulting transcript (concatenating all audio files if audio is not a single file)

        Args:
            audio: filePath, list of filePaths, or path to directory containing all the audio files

        """

        # create temp folder:
        if not os.path.exists(self._tempDir):
            os.mkdir(self._tempDir)

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
        if os.path.exists(self._tempDir):
            shutil.rmtree(self._tempDir)

    def plot(self, audio=None, stopwords=None, max_words=200, background_color="white", colormap="viridis", **kwargs):
        """
        Wrapper of plotWordCloud.
                
        Args:
            audio: filePath, list of filePaths, or path to directory containing all the audio files, [default = None]
            stopwords: list of str, words to remove from the computation of WordCloud
            max_words: max num of words to plot
            background_color: str [default = "white"]
            colormap: matplotlib colormap [default = "viridis"]
            **kwargs: additional arguments of the WordCloud class [*] 

        Returns:
            figure handle
            
        [*] http://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud

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
            
            figureHandle = plotWordCloud(text=self.text, language=self.language, stopwords=stopwords,
                                         max_words=max_words, background_color=background_color, colormap=colormap,
                                         **kwargs)
            return figureHandle

        else:
            raise ValueError("No transcription already stored. Please provide 'audio' input argument")

    def getTranscription(self):
        """
        Returns:
            text: (str) speech-to-text transcript (concatenating all audio files if audio is not a single file)

        """
        warnLog = "No transcription stored. run SpeechCloud.transcribe(audio) first"
        return self.text if self.text else logger.warning(warnLog)

    def saveTranscription(self, tgtFile):
        """
        Save transcription into txt file
        
        Args:
            tgtFile: (str) full path to .txt target file storing the speech-to-text transcription output

        """
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
            _ = extractDesiredAudioSegment(srcFile=srcFilePath, tgtWavFile=tgtFilePath, minutes=self.minutes)
            del _

        else:
            # use all audio file:
            tgtFilePath = srcFilePath

        # create temporary child folder:
        chunksFolder = os.path.join(self._tempDir, "chunks")
        if not os.path.exists(chunksFolder):
            os.mkdir(chunksFolder)

        splitIntoShorterAudioChunks(srcFile=tgtFilePath, tgtDir=chunksFolder, chunkDurationMin=1,
                                    tgtFormat="wav", dBChangeVolume=self.dBChangeVolume)

        text = textFromMultipleSpeechFiles(chunksFolder, language=self.language, engine=self.engine)

        return text


if __name__ == "__main__":


    # TEST SECTION -------------------------------------------------------------------------------------------------

    import sys

    tgtDir = r"C:\Users\RuggeroBETTINARDI\Desktop\temp"
    fileName = "createTempInTranscribe"

    audioGerry = [
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\esposizione.m4a",
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta1.m4a",
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta2.m4a",
    ]
    inputLst = [
        r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\esposizione.m4a",
        # r"P:\WORK\PYTHONPATH\rug\projects\autoradiolockdown\ruggero-dev\autolog\audio\puntate\mp3\segmentsMp3\puntata23\noSong_10.mp3",
        # r"P:\WORK\PYTHONPATH\rug\projects\autoradiolockdown\ruggero-dev\autolog\audio\puntate\mp3\segmentsMp3\puntata23\noSong_3.mp3",
        # r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta1.m4a",
        # r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio\risposta2.m4a",
        # r"P:\WORK\PYTHONPATH\rug\datasets\laureaGerry\audio",  # directory
        # audioGerry,  # list
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

        sc = SpeechCloud(language='it')
        # sc = SpeechCloud(language='it', minutes=[1, 3])

        fig = sc.plot(audio=audio, stopwords=STOPWORDS, max_words=400)
        # sc.plot(audio=audio, stopwords=STOPWORDS, max_words=400, color_func=lambda *args, **kwargs: "white", background_color="red")
        # sc.plot(audio=None, stopwords=STOPWORDS, max_words=400, color_func=lambda *args, **kwargs: "white", background_color="black")
        # sc.plot(audio=audio, stopwords=STOPWORDS, max_words=400, color_func=lambda *args, **kwargs: "white", background_color="green")
        # sc.plot(audio=audio, stopwords=STOPWORDS, max_words=400, colormap="inferno")

        # text = sc.getTranscription()
        # print(text)

        # plt.close("all")

        fig.canvas.start_event_loop(sys.float_info.min)
        plt.savefig(os.path.join(tgtDir, fileName + ".png"), bbox_inches="tight")
        plt.close(fig)

        print(sc.text)

    print("FINISHED")

    # --------------------------------------------------------------------------------------------------------------
