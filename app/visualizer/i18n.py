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
    def prompt_name(self) -> str:
        """English name of the language, for the generator's answer-language prompt."""
        return {
            Language.HEBREW: "Hebrew",
            Language.ARABIC: "Arabic",
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
    GENERATOR = "מנוע התשובה"
    GEN_ECHO = "קטעים שאוחזרו (ללא ניסוח)"
    GEN_CLAUDE = "Claude"
    GEN_GEMINI = "Gemini (חינמי, רוטציית מודלים)"
    GEN_CEREBRAS = "Cerebras (חינמי)"
    GENERATOR_ERROR = "יצירת התשובה נכשלה: {error}"
    # WhatsApp conversation flow.
    SKIP = "דלג"
    ASK_ANOTHER = "אפשר לשאול שאלה נוספת, או להוסיף פרטים על האירוע כדי לדייק את התשובה."
    ADD_DETAILS = "הוספת פרטים"
    EXPAND_PROMPT = "כמה מילים על מה שקרה, ואחזור לתשובה מדויקת יותר."
    INVALID_AGE = "נא להזין גיל תקין (מספר), או ללחוץ על דילוג."
    WELCOME = "שלום! אני הצ'אטבוט של אילאף, ואפשר לשאול אותי על זכויות וסיוע לנפגעי אירועי אלימות."
    CHANGE_LANGUAGE = "לשינוי שפה: לחצו על אחד הכפתורים, או כתבו \"שפה\" בכל שלב."
    RESTART_DONE = "התחלנו מחדש."
    RESTART_HINT = "לאיפוס השיחה בכל שלב: כתבו \"התחל מחדש\"."
    # Reporting a conversation (saved only after an explicit approval).
    REPORT_HINT = "לדיווח על השיחה בכל שלב: כתבו \"דיווח\"."
    REPORT_CONSENT = (
        "דיווח על השיחה פירושו שנשמור עותק שלה באחסון לטווח ארוך, "
        "כדי שצוות אילאף יוכל לקרוא אותה ולטפל בה. "
        "השיחה תישמר רק אם תלחצו על אישור. מספר הטלפון שלכם לא יישמר."
    )
    REPORT_APPROVE = "אישור"
    REPORT_CANCEL = "ביטול"
    REPORT_CANCELLED = "הדיווח בוטל, ולא נשמר דבר."
    REPORT_REASON_PROMPT = (
        "מה תרצו שנדע על השיחה? אפשר לכתוב בכמה מילים, או ללחוץ על דילוג."
    )
    REPORT_CONTACT_PROMPT = (
        "אם תרצו שנחזור אליכם, כתבו טלפון או אימייל. אחרת לחצו על דילוג."
    )
    REPORT_DONE = "הדיווח נשמר. תודה."
    REPORT_FAILED = "שמירת הדיווח נכשלה, ולא נשמר דבר. אפשר לנסות שוב מאוחר יותר."


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
    GENERATOR = "محرك الجواب"
    GEN_ECHO = "المقاطع المسترجعة (بدون صياغة)"
    GEN_CLAUDE = "Claude"
    GEN_GEMINI = "Gemini (مجاني، تدوير موديلات)"
    GEN_CEREBRAS = "Cerebras (مجاني)"
    GENERATOR_ERROR = "فشل توليد الجواب: {error}"
    # WhatsApp conversation flow.
    SKIP = "تخطّي"
    ASK_ANOTHER = "فيكم تسألوا سؤال تاني، أو تضيفوا تفاصيل عن الحادثة لتدقيق الجواب."
    ADD_DETAILS = "إضافة تفاصيل"
    EXPAND_PROMPT = "احكوا بكلمتين شو صار، ورجّع جواب أدقّ."
    INVALID_AGE = "لطفاً دخّلوا عمر صحيح (رقم)، أو اضغطوا تخطّي."
    WELCOME = "مرحبا! أنا الشات بوت تبع إيلاف، وفيكم تسألوني عن الحقوق والمساعدة لمتضرري حوادث العنف."
    CHANGE_LANGUAGE = "لتغيير اللغة: اضغطوا على أحد الأزرار، أو اكتبوا \"لغة\" بأي وقت."
    RESTART_DONE = "بدأنا من جديد."
    RESTART_HINT = "لإعادة المحادثة في أي وقت: اكتبوا \"من البداية\"."
    # Reporting a conversation (saved only after an explicit approval).
    REPORT_HINT = "للتبليغ عن المحادثة بأي وقت: اكتبوا \"تبليغ\"."
    REPORT_CONSENT = (
        "التبليغ عن المحادثة يعني إنّنا رح نحتفظ بنسخة منها بتخزين طويل الأمد، "
        "حتى فريق إيلاف يقدر يقرأها ويتابعها. "
        "بتنحفظ بس إذا ضغطتوا موافقة. رقم تلفونكم ما بينحفظ."
    )
    REPORT_APPROVE = "موافقة"
    REPORT_CANCEL = "إلغاء"
    REPORT_CANCELLED = "التبليغ انلغى، وما انحفظ إشي."
    REPORT_REASON_PROMPT = (
        "شو بدكم نعرف عن المحادثة؟ فيكم تكتبوا بكلمتين، أو تضغطوا تخطّي."
    )
    REPORT_CONTACT_PROMPT = (
        "إذا بدكم نرجع نتواصل معكم، اكتبوا تلفون أو إيميل. غير هيك اضغطوا تخطّي."
    )
    REPORT_DONE = "التبليغ انحفظ. شكراً إلكم."
    REPORT_FAILED = "ما زبط حفظ التبليغ، وما انحفظ إشي. فيكم تجرّبوا بعدين."


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
    GENERATOR = "Answer engine"
    GEN_ECHO = "Retrieved passages (no phrasing)"
    GEN_CLAUDE = "Claude"
    GEN_GEMINI = "Gemini (free, model rotation)"
    GEN_CEREBRAS = "Cerebras (free)"
    GENERATOR_ERROR = "Answer generation failed: {error}"
    # WhatsApp conversation flow.
    SKIP = "Skip"
    ASK_ANOTHER = "You can ask another question, or add details about what happened to sharpen the answer."
    ADD_DETAILS = "Add details"
    EXPAND_PROMPT = "A few words about what happened, and I'll give a more precise answer."
    INVALID_AGE = "Please enter a valid age (a number), or tap Skip."
    WELCOME = "Hi! I'm the Eilaf chatbot — ask me about the rights of, and support for, victims of violent events."
    CHANGE_LANGUAGE = "To change language: tap one of the buttons, or type \"language\" at any point."
    RESTART_DONE = "Started fresh."
    RESTART_HINT = "To restart the conversation at any point, type \"restart\"."
    # Reporting a conversation (saved only after an explicit approval).
    REPORT_HINT = "To report this conversation at any point, type \"report\"."
    REPORT_CONSENT = (
        "Reporting means we keep a copy of this conversation in long-term "
        "storage, so the Eilaf team can read it and follow up. "
        "It is saved only if you tap Approve. Your phone number is not saved."
    )
    REPORT_APPROVE = "Approve"
    REPORT_CANCEL = "Cancel"
    REPORT_CANCELLED = "The report was cancelled, and nothing was saved."
    REPORT_REASON_PROMPT = (
        "What would you like us to know about this conversation? "
        "A few words, or tap Skip."
    )
    REPORT_CONTACT_PROMPT = (
        "If you'd like us to get back to you, type a phone number or an email. "
        "Otherwise tap Skip."
    )
    REPORT_DONE = "The report was saved. Thank you."
    REPORT_FAILED = (
        "The report could not be saved, and nothing was stored. "
        "You can try again later."
    )


TEXTS: dict[Language, type[Enum]] = {
    Language.HEBREW: HebrewText,
    Language.ARABIC: ArabicText,
    Language.ENGLISH: EnglishText,
}
