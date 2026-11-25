"""Generate a 2,400-row synthetic email spam dataset.

Mirrors the structure of `email_spam_detection_synthetic.csv`:
columns subject, body, label (0=legitimate, 1=spam), split
(train 960+960, validation 120+120, test 120+120).

Writing templates are assigned to exactly one split, so no template
crosses the train/validation/test boundary. Templates share realistic
class-level cues (greetings, calls to action, sign-offs), and promotional
words such as "discount" deliberately appear in both classes.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "emails.csv"

FIRST = ["Aarav", "Meera", "Rohan", "Sneha", "Vikram", "Priya", "Karthik", "Ananya",
         "Rahul", "Divya", "Arjun", "Nisha", "Sanjay", "Kavya", "Aditya", "Ishita",
         "Maya", "Daniel", "Sofia", "Liam", "Emma", "Noah", "Olivia", "Ethan"]
LAST = ["Sharma", "Patel", "Iyer", "Nair", "Reddy", "Menon", "Gupta", "Rao",
        "Krishnan", "Bose", "Miller", "Garcia", "Chen", "Wilson", "Brown", "Silva"]
COMPANY = ["Northwind Labs", "Acme Analytics", "Bluewave Systems", "Vertex Retail",
           "Orbit Media", "Sunrise Hospital", "Lakeside University", "Trellis Bank",
           "Kite Logistics", "Mosaic Studio", "Falcon Freight", "GreenLeaf Foods"]
PRODUCT = ["wireless keyboard", "noise-cancelling headphones", "coffee grinder",
           "running shoes", "office chair", "27-inch monitor", "bluetooth speaker",
           "standing desk", "mechanical watch", "robot vacuum"]
CITY = ["Chennai", "Bengaluru", "Mumbai", "Hyderabad", "Coimbatore", "Pune",
        "Delhi", "Kochi", "Austin", "Berlin", "Toronto", "Singapore"]
DOMAIN = ["northwindlabs.com", "acme.io", "vertexretail.com", "orbitmedia.co",
          "kite-logistics.com", "mosaicstudio.net", "greenleaf.example"]
TEAM = ["Accounts", "Support", "Operations", "Academic Office", "Security", "Billing"]


def pick(r, seq):
    return r.choice(seq)


def amount(r, lo, hi):
    return f"{r.randint(lo, hi):,}"


def ref(r):
    return f"{r.choice('ABCDEFGH')}{r.randint(100000, 999999)}"


# ---------------------------------------------------------------- legitimate
def legit_standup(r):
    who = f"{pick(r, FIRST)} {pick(r, LAST)}"
    return (f"Standup notes - {pick(r, ['payments', 'onboarding', 'search', 'billing'])} squad",
            f"Quick summary from today's standup at {pick(r, ['9:30', '10:00', '11:15', '16:00'])}. "
            f"{who} finished the {pick(r, ['API integration', 'dashboard redesign', 'migration script', 'test suite'])} "
            f"and will pick up the {pick(r, ['review comments', 'flaky tests', 'deployment checklist', 'docs update'])} next.\n\n"
            f"The only blocker is {pick(r, ['design sign-off', 'staging access', 'legal review', 'a vendor reply'])}.")


def legit_meeting(r):
    who = f"{pick(r, FIRST)} {pick(r, LAST)}"
    day = pick(r, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    return (f"Invitation: {pick(r, ['Q3 planning', 'roadmap review', 'design critique', 'budget sync'])} on {day}",
            f"You are invited to the {pick(r, ['quarterly planning', 'roadmap review', 'design critique', 'budget sync'])} "
            f"session on {day} at {r.randint(9, 16)}:00 in {pick(r, ['Room 4B', 'the main hall', 'Conference Room 2', 'Google Meet'])}.\n\n"
            f"Agenda: {pick(r, ['hiring plan', 'release scope', 'customer feedback', 'cost review'])} and next steps. "
            f"Please confirm your attendance by replying to this email.\n\n"
            f"Organised by {who}.")


def legit_order(r):
    thing = pick(r, PRODUCT)
    return (f"Your order {ref(r)} has shipped",
            f"Good news - your {thing} is on its way. Order {ref(r)} was dispatched from our "
            f"{pick(r, CITY)} warehouse today and should arrive in {r.randint(2, 6)} business days.\n\n"
            f"You can track the parcel from your account at any time. If anything looks wrong, "
            f"reply to this message and our support team will help.")


def legit_newsletter_discount(r):
    code = f"{pick(r, ['WELCOME', 'THANKS', 'MEMBER', 'SEASON'])}{r.randint(10, 40)}"
    return ("Here is the discount code you requested",
            f"Thanks for subscribing to the {pick(r, COMPANY)} newsletter. As promised, here is the "
            f"{r.randint(10, 30)}% discount code you asked for: {code}.\n\n"
            f"The code is valid on your next order until {r.randint(1, 28)} "
            f"{pick(r, ['January', 'March', 'June', 'September', 'November'])}. "
            f"We send occasional updates about new arrivals, never more than twice a month, and you can "
            f"unsubscribe from the footer of any email at any time.")


def legit_password_reset(r):
    return (f"Reset your {pick(r, COMPANY)} password",
            f"We received a request to reset the password for your {pick(r, COMPANY)} account. "
            f"If this was you, open the link below within 30 minutes:\n\n"
            f"https://{pick(r, DOMAIN)}/reset/{ref(r).lower()}\n\n"
            f"If you did not request a reset, you can safely ignore this email. Your password will not "
            f"change until the link is used. Contact support if you need help.")


def legit_invoice(r):
    return (f"Invoice {ref(r)} from {pick(r, COMPANY)}",
            f"Please find attached invoice {ref(r)} for "
            f"{pick(r, ['consulting services', 'software licences', 'office supplies', 'maintenance'])} "
            f"for the period ending {r.randint(1, 28)}/{r.randint(1, 12)}/2025.\n\n"
            f"Amount due: Rs {amount(r, 4000, 90000)}. Payment terms: "
            f"{pick(r, ['15 days', '30 days', 'on receipt'])}. Let us know if you need a purchase order "
            f"number added to the document.")


def legit_document_reply(r):
    who = f"{pick(r, FIRST)} {pick(r, LAST)}"
    return (f"Re: {pick(r, ['draft report', 'meeting minutes', 'proposal v2', 'budget sheet'])}",
            f"Thanks for sending the {pick(r, ['draft report', 'meeting minutes', 'proposal', 'budget sheet'])}. "
            f"I read through it and left a few comments in the {pick(r, ['shared doc', 'PDF', 'spreadsheet'])}.\n\n"
            f"The {pick(r, ['timeline', 'cost estimate', 'scope', 'owner list'])} needs one more pass. "
            f"Can we close that out before {pick(r, ['Friday', 'the client call', 'the end of the month'])}? "
            f"Happy to jump on a call if that is easier.\n\n"
            f"{who}")


def legit_campus(r):
    return (f"{pick(r, ['Semester registration', 'Exam timetable', 'Lab allocation', 'Fee receipt'])} - {pick(r, COMPANY)}",
            f"This is a reminder that {pick(r, ['semester registration', 'exam timetable publication', 'lab allocation', 'fee payment'])} "
            f"closes on {r.randint(1, 28)}/{r.randint(1, 12)}/2025.\n\n"
            f"Please log in to the student portal and complete the required steps. For questions, contact "
            f"the {pick(r, ['academic office', 'exam cell', 'accounts desk', 'lab coordinator'])}. "
            f"We are happy to help if anything is unclear.")


def legit_bank_alert(r):
    return (f"Transaction alert for your account ending {r.randint(1000, 9999)}",
            f"A transaction of Rs {amount(r, 200, 15000)} was made on your "
            f"{pick(r, ['savings', 'current', 'credit card'])} account ending {r.randint(1000, 9999)} "
            f"at {pick(r, ['a POS terminal', 'an ATM', 'an online merchant'])} in {pick(r, CITY)}.\n\n"
            f"If this was you, no action is needed. If not, call the number on the back of your card "
            f"immediately. We will never ask for your PIN or password by email.")


def legit_appointment(r):
    return (f"Appointment reminder - {r.randint(1, 28)}/{r.randint(1, 12)}/2025",
            f"This is a reminder of your appointment with Dr. {pick(r, LAST)} on "
            f"{r.randint(1, 28)}/{r.randint(1, 12)}/2025 at {r.randint(9, 17)}:{pick(r, ['00', '15', '30', '45'])}.\n\n"
            f"Please arrive 10 minutes early and bring your previous reports. To reschedule, call the front "
            f"desk at {r.randint(4000000, 4999999)}. We look forward to seeing you.")


def legit_github(r):
    who = f"{pick(r, FIRST)} {pick(r, LAST)}".lower().replace(" ", "")
    return (f"[{pick(r, ['acme', 'orbit', 'vertex'])}/api] Pull request #{r.randint(100, 999)}: "
            f"{pick(r, ['fix retry loop', 'add pagination', 'tighten validation', 'drop unused dep'])}",
            f"{who} opened a pull request.\n\n"
            f"{pick(r, ['This fixes', 'This adds', 'This removes'])} the "
            f"{pick(r, ['retry loop', 'pagination helper', 'input validation', 'legacy dependency'])} and "
            f"includes {r.randint(1, 6)} new tests.\n\n"
            f"Reviewers: {pick(r, FIRST)}, {pick(r, FIRST)}. CI: {pick(r, ['passing', '2 checks passed'])}. "
            f"View the full diff on GitHub.")


def legit_flight(r):
    return (f"Booking confirmed: {pick(r, CITY)} to {pick(r, CITY)}",
            f"Your booking {ref(r)} is confirmed.\n\n"
            f"Flight {pick(r, ['AI', '6E', 'EK', 'LH'])}{r.randint(100, 999)} departs {pick(r, CITY)} at "
            f"{r.randint(5, 21)}:{pick(r, ['05', '20', '45'])} and arrives {pick(r, CITY)} at "
            f"{r.randint(6, 23)}:{pick(r, ['10', '35', '55'])}.\n\n"
            f"Check-in opens 48 hours before departure. Baggage allowance is "
            f"{pick(r, ['15 kg', '20 kg', '25 kg'])}. Have a good trip.")


# --------------------------------------------------------------------- spam
def spam_discount(r):
    return (f"{r.randint(50, 90)}% OFF - today only, {pick(r, FIRST)}!",
            f"We are clearing stock and you have been selected for an exclusive {r.randint(50, 90)}% discount "
            f"on {pick(r, PRODUCT)} and hundreds of other items. No code needed and free shipping on every order.")


def spam_lottery(r):
    return ("Congratulations! You have won a cash prize",
            f"Your email address was randomly selected in our international draw and you have won "
            f"{pick(r, ['Rs 25,00,000', 'a 7-day holiday', 'a new phone', 'a Rs 50,000 gift card'])}.\n\n"
            f"To release your prize, send your full name, address and a copy of your ID to our claims agent. "
            f"A small processing fee of Rs {amount(r, 500, 5000)} is required.")


def spam_phish_account(r):
    return (f"Urgent: your {pick(r, ['bank', 'paypal', 'amazon', 'netflix'])} account will be suspended",
            f"We detected unusual activity on your account, so access has been limited for your security. "
            f"Verify your identity and confirm your password within 24 hours to restore access.\n\n"
            f"Enter your login details on our secure page: "
            f"http://{pick(r, ['secure-login', 'account-verify', 'billing-update'])}-{ref(r).lower()}.{pick(r, ['com', 'net', 'info'])}\n\n"
            f"If you do not confirm your details, your account will be permanently closed and any balance lost.")


def spam_invoice(r):
    return (f"FINAL NOTICE: unpaid invoice {ref(r)}",
            f"Our records show invoice {ref(r)} for Rs {amount(r, 3000, 45000)} remains unpaid. "
            f"Despite several reminders, no payment has been received.\n\n"
            f"Open the attached document and pay immediately to avoid legal action. If payment is not "
            f"received within 48 hours, your case will be forwarded to our collections partner.")


def spam_crypto(r):
    return (f"Turn Rs {amount(r, 5000, 20000)} into Rs {amount(r, 100000, 900000)} in 30 days",
            f"Our automated trading bot has returned {r.randint(300, 900)}% this year and members are "
            f"withdrawing profits every week.\n\n"
            f"Join the private group today and receive a {r.randint(20, 60)}% bonus on your first deposit. "
            f"Past performance guarantees future results.")


def spam_weightloss(r):
    return (f"Lose {r.randint(5, 20)} kg in {r.randint(2, 4)} weeks without dieting",
            f"Doctors are shocked by this simple {pick(r, ['tea', 'pill', 'patch', 'shot'])} that melts fat "
            f"while you sleep. No exercise needed and no diet needed.\n\n"
            f"Order now and get a free bottle with your first purchase. A money back guarantee is included.")


def spam_workhome(r):
    return (f"Earn Rs {amount(r, 2000, 9000)} per day from home - no experience",
            f"We are hiring {r.randint(50, 500)} people to do simple online work from home. Just "
            f"{r.randint(1, 3)} hours a day and you can earn Rs {amount(r, 2000, 9000)} daily.\n\n"
            f"No interview and no skills required. Payment is released every evening. To start, send a "
            f"one-time registration fee of Rs {amount(r, 200, 1500)} and your bank details.")


def spam_delivery(r):
    return (f"Delivery failed: parcel {ref(r)} could not be delivered",
            f"We attempted to deliver your parcel today but no one was available. A redelivery fee of "
            f"Rs {r.randint(25, 150)} is now due.\n\n"
            f"Pay the fee and reschedule here: "
            f"http://{pick(r, ['parcel-track', 'delivery-fix', 'courier-pay'])}.{pick(r, ['xyz', 'top', 'site'])}/{ref(r).lower()}\n\n"
            f"If the fee is not paid within 24 hours the parcel will be returned to the sender.")


def spam_dating(r):
    return (f"{pick(r, FIRST)} sent you a message",
            f"I came across your profile and thought you looked interesting. I am {r.randint(22, 35)} and "
            f"live nearby. I am online now and would love to chat.\n\n"
            f"Reply to this email or visit my page: "
            f"http://{pick(r, ['meet-now', 'chat-love', 'single-local'])}.{pick(r, ['xyz', 'club', 'date'])}/{ref(r).lower()}\n\n"
            f"Send me your number and we can meet tonight.")


def spam_bank_freeze(r):
    return ("Your debit card has been blocked",
            f"We blocked your debit card ending {r.randint(1000, 9999)} after {r.randint(3, 9)} failed "
            f"verification attempts.\n\n"
            f"To unblock it, confirm your card number, CVV and PIN on our secure page: "
            f"http://{pick(r, ['bank-secure', 'card-verify', 'netbanking-update'])}.{pick(r, ['xyz', 'info', 'top'])}/{ref(r).lower()}\n\n"
            f"If not confirmed within 12 hours the card will be cancelled permanently.")


def spam_subscription(r):
    return (f"Your subscription auto-renewed for Rs {amount(r, 999, 8999)}",
            f"Your subscription has been automatically renewed for Rs {amount(r, 999, 8999)}.\n\n"
            f"If you did not authorise this charge, you are entitled to a refund. Claim it here within "
            f"24 hours: http://{pick(r, ['refund-now', 'cancel-charge', 'billing-help'])}.{pick(r, ['xyz', 'site', 'top'])}/{ref(r).lower()}\n\n"
            f"Failure to claim means you accept the charge.")


def spam_seo(r):
    return ("Quick question about your website",
            f"I found your website and noticed it is not ranking on page one for "
            f"{pick(r, ['your main keywords', 'local searches', 'high-intent terms', 'mobile queries'])}.\n\n"
            f"We can get you {r.randint(3, 20)}x more traffic in {r.randint(30, 90)} days with our proven "
            f"SEO package. The first {r.randint(5, 15)} links are free.\n\n"
            f"Reply with a good time for a {r.randint(10, 30)}-minute call and we will send a free audit.")


LEGIT = [legit_standup, legit_meeting, legit_order, legit_newsletter_discount,
         legit_password_reset, legit_invoice, legit_document_reply, legit_campus,
         legit_bank_alert, legit_appointment, legit_github, legit_flight]
SPAM = [spam_discount, spam_lottery, spam_phish_account, spam_invoice,
        spam_crypto, spam_weightloss, spam_workhome, spam_delivery,
        spam_dating, spam_bank_freeze, spam_subscription, spam_seo]

SPLITS = {"train": (0, 4), "validation": (4, 8), "test": (8, 12)}
PER_SPLIT_PER_CLASS = {"train": 960, "validation": 120, "test": 120}

LEGIT_GREET = ["Hi {first},", "Hello {first},", "Dear {First} {Last},", "Hi {first},", "Hello {first},"]
LEGIT_CLOSE = [
    "Let us know if you have any questions.",
    "Please reply if anything looks wrong.",
    "No action is needed if this was you.",
    "Happy to help if you need anything else.",
    "Get in touch if you would like to discuss it further.",
]
LEGIT_SIGN = [
    "Thanks,\n{Name}\n{Company}",
    "Best regards,\n{Name}\n{Company}",
    "Regards,\n{Team}, {Company}",
    "Thanks,\n{Name}",
    "Kind regards,\n{Name}\n{Company}",
    "{Company}\nThis is an automated message; please do not reply.",
]
SPAM_GREET = ["Dear customer,", "Dear user,", "Dear friend,", "Hello,", "Hi there,", "Dear subscriber,", "Dear winner,"]
SPAM_URGENT = {
    "promo": ["This offer expires in {n} hours.", "Only {n} left - act now!", "Hurry, the sale ends tonight.",
              "This is a limited time offer.", "Claim your free gift before it is gone."],
    "account": ["Confirm your details within 24 hours.", "Your account will be closed if you do not respond.",
                "Verify your identity now to avoid suspension.", "This is your final warning."],
    "reply": ["Reply YES and our team will contact you.", "Send your number today.", "Reply now - spots are filling fast.",
              "Do not miss this chance."],
    "prize": ["Claim within {n} hours or the prize is forfeited.", "Your prize is reserved for 24 hours only.",
              "Send your details today to release the prize."],
    "invoice": ["Pay immediately to avoid legal action.", "This is a final notice before collections.",
                "Failure to pay will result in account suspension."],
}
SPAM_CTA = [
    "Click here to claim: http://{a}.{b}/{c}",
    "Visit our secure page: http://{a}.{b}/{c}",
    "Register here: http://{a}.{b}/{c}",
    "Confirm here: http://{a}.{b}/{c}",
]
SPAM_TAIL = ["Guaranteed results.", "100% free to join.", "No experience needed.", "Act now!",
             "Risk-free and fully guaranteed.", "Limited seats available."]
SPAM_SIGN = ["Customer Support", "Security Team", "Billing Department", "HR Team",
             "Account Services", "Prize Department", "The Claims Team"]


def legit_text(r, subject, core):
    first, name = pick(r, FIRST), f"{pick(r, FIRST)} {pick(r, LAST)}"
    greet = pick(r, LEGIT_GREET).format(first=first, First=name.split()[0], Last=name.split()[1])
    sign = pick(r, LEGIT_SIGN).format(Name=name, Company=pick(r, COMPANY), Team=pick(r, TEAM))
    return subject, f"{greet}\n\n{core}\n\n{pick(r, LEGIT_CLOSE)}\n\n{sign}"


def spam_text(r, subject, core, kind="promo"):
    greet = pick(r, SPAM_GREET)
    urgent = pick(r, SPAM_URGENT.get(kind, SPAM_URGENT["promo"])).format(n=r.randint(2, 24))
    cta = pick(r, SPAM_CTA).format(a=pick(r, ["deals-now", "secure-login", "claim-prize", "fast-cash", "verify-now", "parcel-track"]),
                                   b=pick(r, ["xyz", "top", "info", "site", "club"]), c=ref(r).lower())
    tail = pick(r, SPAM_TAIL)
    return subject, f"{greet}\n\n{core}\n\n{urgent} {tail}\n\n{cta}\n\n{pick(r, SPAM_SIGN)}"


SPAM_KIND = {spam_discount: "promo", spam_lottery: "prize", spam_phish_account: "account",
             spam_invoice: "invoice", spam_crypto: "promo", spam_weightloss: "promo",
             spam_workhome: "reply", spam_delivery: "account", spam_dating: "reply",
             spam_bank_freeze: "account", spam_subscription: "invoice", spam_seo: "reply"}


def soft_spam(r, subject, core):
    """Cold outreach that avoids the usual spam furniture: some of these look legitimate."""
    first, name = pick(r, FIRST), f"{pick(r, FIRST)} {pick(r, LAST)}"
    return subject, f"Hi {first},\n\n{core}\n\nBest,\n{name}\n{pick(r, COMPANY)}"


PROMO_LINE = [
    "As a subscriber you also get a 15% discount on your next order this week.",
    "By the way, the seasonal sale ends Friday and members get free shipping.",
    "Your loyalty discount is still available in the offers section of your account.",
]


def build():
    r = random.Random(20261125)
    rows = []
    for split, (lo, hi) in SPLITS.items():
        n = PER_SPLIT_PER_CLASS[split]
        for label, templates in ((0, LEGIT), (1, SPAM)):
            chosen = templates[lo:hi]
            per = n // len(chosen)
            for t in chosen:
                for _ in range(per):
                    subject, core = t(r)
                    if label == 0:
                        subject, body = legit_text(r, subject, core)
                        if r.random() < 0.08:
                            body = body.replace("\n\n", f"\n\n{pick(r, PROMO_LINE)}\n\n", 1)
                    else:
                        subject, body = spam_text(r, subject, core, SPAM_KIND[t])
                        if r.random() < 0.10:
                            subject, body = soft_spam(r, subject, core)
                    rows.append(dict(subject=subject, body=body, label=label, split=split))
    r.shuffle(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "body", "label", "split"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"wrote {len(rows)} rows -> {OUT}")
    print("split/label:", sorted(Counter((x["split"], x["label"]) for x in rows).items()))
    for s in SPLITS:
        print(f"  {s}: {sum(1 for x in rows if x['split'] == s)}")


if __name__ == "__main__":
    build()
