"""UI text for the Streamlit visualizer.

Only the strings the visualizer alone shows live here; the rest (the query
prompt, answer, sources …) are shared with the chatbot and come from
``app.i18n``. ``TEXTS[language]`` looks a name up here first and falls back to
the shared Enum, so the app reads e.g. ``TEXTS[language].TITLE.value`` either way.
"""

from enum import Enum

from app.i18n import TEXTS as SHARED_TEXTS
from app.i18n import Language


class HebrewText(Enum):
    TITLE = "צ'אטבוט מתנדבי אילאף"
    NO_DOCUMENTS = "לא נמצאו מסמכים בתיקיית data/static. הוסיפו קובץ ‎.pdf‏ או ‎.txt‏."
    CONTEXT_WEIGHT = "משקל תיאור האירוע באחזור"
    SIBLING_K = "מספר סעיפים שכנים לכל תוצאה"
    FILTER_EXPANDER = "סינון לפי פרטי הפונה (אופציונלי)"
    SPECIFY_AGE = "ציון גיל"
    GENDER_UNSPECIFIED = "לא צוין"
    LOCALITY = "יישוב"
    LOCALITY_UNSPECIFIED = "לא צוין"
    LANGUAGE_LABEL = "שפה"
    GENERATOR = "מנוע התשובה"
    GEN_ECHO = "קטעים שאוחזרו (ללא ניסוח)"
    GEN_CLAUDE = "Claude"
    GEN_BEDROCK = "Claude (Bedrock)"
    GEN_GEMINI = "Gemini (חינמי, רוטציית מודלים)"
    GEN_CEREBRAS = "Cerebras (חינמי)"


class ArabicText(Enum):
    # Levantine (Palestinian) colloquial Arabic.
    TITLE = "شات بوت متطوعين إيلاف"
    NO_DOCUMENTS = "ما في ملفات بمجلد data/static. ضيفوا ملف ‎.pdf‏ أو ‎.txt‏."
    CONTEXT_WEIGHT = "وزن وصف الحادثة بالاسترجاع"
    SIBLING_K = "عدد البنود المجاورة لكل نتيجة"
    FILTER_EXPANDER = "تصفية حسب معلومات المتوجّه (اختياري)"
    SPECIFY_AGE = "تحديد العمر"
    GENDER_UNSPECIFIED = "غير محدد"
    LOCALITY = "البلد"
    LOCALITY_UNSPECIFIED = "غير محدد"
    LANGUAGE_LABEL = "اللغة"
    GENERATOR = "محرك الجواب"
    GEN_ECHO = "المقاطع المسترجعة (بدون صياغة)"
    GEN_CLAUDE = "Claude"
    GEN_BEDROCK = "Claude (Bedrock)"
    GEN_GEMINI = "Gemini (مجاني، تدوير موديلات)"
    GEN_CEREBRAS = "Cerebras (مجاني)"


class EnglishText(Enum):
    TITLE = "Eilaf Volunteers Chatbot"
    NO_DOCUMENTS = "No documents found in data/static. Add a .pdf or .txt file."
    CONTEXT_WEIGHT = "Event-description weight in retrieval"
    SIBLING_K = "Neighbouring clauses per result"
    FILTER_EXPANDER = "Filter by applicant details (optional)"
    SPECIFY_AGE = "Specify age"
    GENDER_UNSPECIFIED = "Unspecified"
    LOCALITY = "Locality"
    LOCALITY_UNSPECIFIED = "Unspecified"
    LANGUAGE_LABEL = "Language"
    GENERATOR = "Answer engine"
    GEN_ECHO = "Retrieved passages (no phrasing)"
    GEN_CLAUDE = "Claude"
    GEN_BEDROCK = "Claude (Bedrock)"
    GEN_GEMINI = "Gemini (free, model rotation)"
    GEN_CEREBRAS = "Cerebras (free)"


class _Texts:
    """One language's visualizer strings, falling back to the shared ones."""

    def __init__(self, *enums: type[Enum]) -> None:
        self._enums = enums

    def __getattr__(self, name: str) -> Enum:
        for enum in self._enums:
            if name in enum.__members__:
                return enum[name]
        raise AttributeError(name)


TEXTS: dict[Language, _Texts] = {
    language: _Texts(own, SHARED_TEXTS[language])
    for language, own in (
        (Language.HEBREW, HebrewText),
        (Language.ARABIC, ArabicText),
        (Language.ENGLISH, EnglishText),
    )
}
