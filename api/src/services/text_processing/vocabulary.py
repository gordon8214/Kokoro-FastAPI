import json
from functools import lru_cache
from pathlib import Path


def get_vocab():
    """Get the vocabulary dictionary mapping characters to token IDs"""
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    # Create vocabulary dictionary
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    return {symbol: i for i, symbol in enumerate(symbols)}


# Initialize vocabulary
VOCAB = get_vocab()


@lru_cache(maxsize=1)
def get_kokoro_vocab() -> dict[str, int]:
    """Load the exact vocabulary shipped with the active Kokoro 1.x model."""
    source_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        Path("models/v1_0/config.json"),
        Path("builds/v1_0/config.json"),
    ):
        config_path = source_root / relative_path
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            return {
                str(symbol): int(token_id)
                for symbol, token_id in payload["vocab"].items()
            }
    raise FileNotFoundError("Kokoro model vocabulary config was not found")


def tokenize_kokoro(phonemes: str) -> list[int]:
    """Convert phonemes with the exact vocabulary used by Kokoro 1.x."""
    vocab = get_kokoro_vocab()
    unknown = sorted(set(phonemes).difference(vocab))
    if unknown:
        raise ValueError(f"Kokoro vocabulary does not contain symbols: {unknown}")
    return [vocab[phoneme] for phoneme in phonemes]


def tokenize(phonemes: str) -> list[int]:
    """Convert phonemes string to token IDs

    Args:
        phonemes: String of phonemes to tokenize

    Returns:
        List of token IDs
    """
    # Strip phonemes to remove leading/trailing spaces that could cause artifacts
    phonemes = phonemes.strip()
    return [i for i in map(VOCAB.get, phonemes) if i is not None]


def decode_tokens(tokens: list[int]) -> str:
    """Convert token IDs back to phonemes string

    Args:
        tokens: List of token IDs

    Returns:
        String of phonemes
    """
    # Create reverse mapping
    id_to_symbol = {i: s for s, i in VOCAB.items()}
    return "".join(id_to_symbol[t] for t in tokens)
