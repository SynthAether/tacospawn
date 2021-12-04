class Config:
    """Model configuration.
    """
    def __init__(self, vocabs: int):
        """Initializer.
        Args:
            vocabs: the size of the dictionary.
        """
        self.vocabs = vocabs

        # channel info
        self.embeddings = 256
        self.channels = 128

        # encoder
        self.enc_prenet = [256]
        self.enc_dropout = 0.5

        # cbhg
        self.cbhg_banks = 16
        self.cbhg_pool = 2
        self.cbhg_kernels = 3
        self.cbhg_highways = 4

        # decoder
        self.reduction = 2

        self.dec_prenet = [256]
        self.dec_dropout = 0.5
