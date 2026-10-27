"""User-facing text for the chatbot, in every supported language.

Each supported language keeps its strings in its own ``Enum`` whose member
names are identical across languages. The bot selects one Enum via
``TEXTS[language]`` and reads e.g. ``TEXTS[language].WELCOME.value``.

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
    QUERY_PROMPT = "מה תרצו לשאול?"
    CONTEXT_PROMPT = "תיאור קצר של אירוע האלימות (אופציונלי)"
    AGE = "גיל"
    GENDER = "מגדר"
    GENDER_FEMALE = "נקבה"
    GENDER_MALE = "זכר"
    ANSWER = "תשובה"
    NO_ANSWER = "אין תשובה."
    SOURCES = "מקורות"
    GENERATOR_ERROR = "יצירת התשובה נכשלה: {error}"
    # Conversation flow.
    SKIP = "דלג"
    ASK_ANOTHER = "אפשר לשאול שאלה נוספת, או להוסיף פרטים על האירוע כדי לדייק את התשובה."
    ADD_DETAILS = "הוספת פרטים"
    SHOW_SOURCES = "הצגת מקורות"
    SOURCES_UNAVAILABLE = "המקורות של התשובה הזו כבר אינם זמינים."
    EXPAND_PROMPT = "כמה מילים על מה שקרה, ואחזור לתשובה מדויקת יותר."
    INVALID_AGE = "נא להזין גיל תקין (מספר), או ללחוץ על דילוג."
    WELCOME = "שלום! אני הצ'אטבוט של אילאף, ואפשר לשאול אותי על זכויות וסיוע לנפגעי אירועי אלימות."
    WELCOME_BACK = "ברוכים השבים"
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
    # Shown first, and only under EILAF_ENV=dev.
    DEV_NOTICE = (
        "הודעה חשובה: זו גרסת בדיקה. כל השיחה נשמרת — כולל כל מה שתכתבו — "
        "כדי שהצוות יוכל לפתח את השירות."
    )


class ArabicText(Enum):
    # Levantine (Palestinian) colloquial Arabic.
    QUERY_PROMPT = "شو بدكم تسألوا؟"
    CONTEXT_PROMPT = "وصف مختصر لحادثة العنف (اختياري)"
    AGE = "العمر"
    GENDER = "الجنس"
    GENDER_FEMALE = "أنثى"
    GENDER_MALE = "ذكر"
    ANSWER = "الجواب"
    NO_ANSWER = "ما في جواب."
    SOURCES = "المصادر"
    GENERATOR_ERROR = "فشل توليد الجواب: {error}"
    # Conversation flow.
    SKIP = "تخطّي"
    ASK_ANOTHER = "فيكم تسألوا سؤال تاني، أو تضيفوا تفاصيل عن الحادثة لتدقيق الجواب."
    ADD_DETAILS = "إضافة تفاصيل"
    SHOW_SOURCES = "عرض المصادر"
    SOURCES_UNAVAILABLE = "مصادر هالجواب ما عادت متوفرة."
    EXPAND_PROMPT = "احكوا بكلمتين شو صار، ورجّع جواب أدقّ."
    INVALID_AGE = "لطفاً دخّلوا عمر صحيح (رقم)، أو اضغطوا تخطّي."
    WELCOME = "مرحبا! أنا الشات بوت تبع إيلاف، وفيكم تسألوني عن الحقوق والمساعدة لمتضرري حوادث العنف."
    WELCOME_BACK = "أهلا فيكم"
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
    # Shown first, and only under EILAF_ENV=dev.
    DEV_NOTICE = (
        "ملاحظة مهمة: هاي نسخة تجريبية. كل المحادثة بتنحفظ — بما فيها كل اللي "
        "بتكتبوه — حتى الفريق يقدر يطوّر الخدمة."
    )


class EnglishText(Enum):
    QUERY_PROMPT = "What would you like to ask?"
    CONTEXT_PROMPT = "Brief description of the violent event (optional)"
    AGE = "Age"
    GENDER = "Gender"
    GENDER_FEMALE = "Female"
    GENDER_MALE = "Male"
    ANSWER = "Answer"
    NO_ANSWER = "No answer."
    SOURCES = "Sources"
    GENERATOR_ERROR = "Answer generation failed: {error}"
    # Conversation flow.
    SKIP = "Skip"
    ASK_ANOTHER = "You can ask another question, or add details about what happened to sharpen the answer."
    ADD_DETAILS = "Add details"
    SHOW_SOURCES = "Show sources"
    SOURCES_UNAVAILABLE = "The sources for that answer are no longer available."
    EXPAND_PROMPT = "A few words about what happened, and I'll give a more precise answer."
    INVALID_AGE = "Please enter a valid age (a number), or tap Skip."
    WELCOME = "Hi! I'm the Eilaf chatbot — ask me about the rights of, and support for, victims of violent events."
    WELCOME_BACK = "Welcome back!"
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
    # Shown first, and only under EILAF_ENV=dev.
    DEV_NOTICE = (
        "Please note: this is a test version. The whole conversation is saved, "
        "including everything you write, so the team can develop the service."
    )


TEXTS: dict[Language, type[Enum]] = {
    Language.HEBREW: HebrewText,
    Language.ARABIC: ArabicText,
    Language.ENGLISH: EnglishText,
}
