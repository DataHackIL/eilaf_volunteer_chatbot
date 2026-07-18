"""UI text translations for the Streamlit visualizer.

Each supported language keeps its strings in its own ``Enum`` whose member
names are identical across languages. The app selects one Enum via
``TEXTS[language]`` and reads e.g. ``TEXTS[language].TITLE.value``.

Arabic uses a Levantine (Palestinian/Levant) colloquial register rather than
Modern Standard Arabic, to match how volunteers and applicants actually speak.
"""

from enum import Enum


class Language(Enum):
    """Supported UI languages."""

    HEBREW = "he"
    ARABIC = "ar"
    ENGLISH = "en"

    @property
    def native_name(self) -> str:
        return {
            Language.HEBREW: "עברית",
            Language.ARABIC: "العربية",
            Language.ENGLISH: "English",
        }[self]

    @property
    def is_rtl(self) -> bool:
        return self in (Language.HEBREW, Language.ARABIC)


class HebrewText(Enum):
    TITLE = "צ'אטבוט מתנדבי אילאף"
    NO_DOCUMENTS = "לא נמצאו מסמכים בתיקיית data/static. הוסיפו קובץ ‎.pdf‏ או ‎.txt‏."
    QUERY_PROMPT = "מה תרצו לשאול?"
    CONTEXT_PROMPT = "תיאור קצר של אירוע האלימות (אופציונלי)"
    CONTEXT_WEIGHT = "משקל תיאור האירוע באחזור"
    SIBLING_K = "מספר סעיפים שכנים לכל תוצאה"
    FILTER_EXPANDER = "סינון לפי פרטי הפונה (אופציונלי)"
    SPECIFY_AGE = "ציון גיל"
    AGE = "גיל"
    GENDER = "מגדר"
    GENDER_UNSPECIFIED = "לא צוין"
    GENDER_FEMALE = "נקבה"
    GENDER_MALE = "זכר"
    LOCALITY = "יישוב"
    LOCALITY_UNSPECIFIED = "לא צוין"
    ANSWER = "תשובה"
    NO_ANSWER = "אין תשובה."
    SOURCES = "מקורות"
    LANGUAGE_LABEL = "שפה"


class ArabicText(Enum):
    # Levantine (Palestinian) colloquial Arabic.
    TITLE = "شات بوت متطوعين إيلاف"
    NO_DOCUMENTS = "ما في ملفات بمجلد data/static. ضيفوا ملف ‎.pdf‏ أو ‎.txt‏."
    QUERY_PROMPT = "شو بدكم تسألوا؟"
    CONTEXT_PROMPT = "وصف مختصر لحادثة العنف (اختياري)"
    CONTEXT_WEIGHT = "وزن وصف الحادثة بالاسترجاع"
    SIBLING_K = "عدد البنود المجاورة لكل نتيجة"
    FILTER_EXPANDER = "تصفية حسب معلومات المتوجّه (اختياري)"
    SPECIFY_AGE = "تحديد العمر"
    AGE = "العمر"
    GENDER = "الجنس"
    GENDER_UNSPECIFIED = "غير محدد"
    GENDER_FEMALE = "أنثى"
    GENDER_MALE = "ذكر"
    LOCALITY = "البلد"
    LOCALITY_UNSPECIFIED = "غير محدد"
    ANSWER = "الجواب"
    NO_ANSWER = "ما في جواب."
    SOURCES = "المصادر"
    LANGUAGE_LABEL = "اللغة"


class EnglishText(Enum):
    TITLE = "Eilaf Volunteers Chatbot"
    NO_DOCUMENTS = "No documents found in data/static. Add a .pdf or .txt file."
    QUERY_PROMPT = "What would you like to ask?"
    CONTEXT_PROMPT = "Brief description of the violent event (optional)"
    CONTEXT_WEIGHT = "Event-description weight in retrieval"
    SIBLING_K = "Neighbouring clauses per result"
    FILTER_EXPANDER = "Filter by applicant details (optional)"
    SPECIFY_AGE = "Specify age"
    AGE = "Age"
    GENDER = "Gender"
    GENDER_UNSPECIFIED = "Unspecified"
    GENDER_FEMALE = "Female"
    GENDER_MALE = "Male"
    LOCALITY = "Locality"
    LOCALITY_UNSPECIFIED = "Unspecified"
    ANSWER = "Answer"
    NO_ANSWER = "No answer."
    SOURCES = "Sources"
    LANGUAGE_LABEL = "Language"


TEXTS: dict[Language, type[Enum]] = {
    Language.HEBREW: HebrewText,
    Language.ARABIC: ArabicText,
    Language.ENGLISH: EnglishText,
}
