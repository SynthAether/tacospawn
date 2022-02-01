class Config:
    """Model configuration.
    """
    def __init__(self, vocabs: int, mel: int, speakers: int):
        """Initializer.
        Args:
            vocabs: the size of the dictionary.
            mel: the number of the mel-scale filterbank bins.
            speakers: the number of the training speakers.
        """
        self.vocabs = vocabs
        self.mel = mel
        self.speakers = speakers

        # channel info
        self.embeddings = 256
        self.channels = 256

        # speaker embedding
        self.modal = 5
        self.spkembed = 128

        # encoder
        self.enc_prenet = [256]
        self.enc_dropout = 0.5

        # cbhg
        self.cbhg_banks = 16
        self.cbhg_pool = 2
        self.cbhg_kernels = 3
        self.cbhg_highways = 4

        # reduction
        self.reduction = 2

        # decoder
        self.dec_prenet = [256, 128]
        self.dec_dropout = 0.5
        self.dec_layers = 2

        # inference
        self.dec_max_factor = 8 // self.reduction
