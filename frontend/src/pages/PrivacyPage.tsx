import { Link } from 'react-router-dom'
import { useI18n } from '../i18n'
import { operatorPublished, privacyContact, privacyOperator, PRIVACY_VERSION } from '../privacy'
import { Icon } from '../ui'

export default function PrivacyPage() {
  const { locale } = useI18n()
  const tr = (en: string, bn: string) => locale === 'bn' ? bn : en
  const sections = [
    {
      title: tr('What is collected', 'কোন তথ্য সংগ্রহ করা হয়'),
      body: tr('We store the complaint you submit, its detected or selected language, optional area and optional phone/email, a random tracking ID, timestamps, analysis, status and related-report grouping. Staff decisions and reasons are recorded. Avoid names, identity documents, passwords, medical details or other people’s private information in complaint text.', 'আপনার অভিযোগের লেখা, শনাক্ত বা নির্বাচিত ভাষা, ঐচ্ছিক এলাকা ও ফোন/ইমেইল, এলোমেলো ট্র্যাকিং আইডি, সময়, বিশ্লেষণ, অবস্থা ও সম্পর্কিত অভিযোগের দল সংরক্ষিত হয়। কর্মকর্তাদের সিদ্ধান্ত ও কারণও রাখা হয়। অভিযোগে নাম, পরিচয়পত্র, পাসওয়ার্ড, চিকিৎসার তথ্য বা অন্যের ব্যক্তিগত তথ্য দেবেন না।'),
    },
    {
      title: tr('Purpose and access', 'ব্যবহারের উদ্দেশ্য ও প্রবেশাধিকার'),
      body: tr('Authorised staff use reports to assess service issues, review suggested priority and identify related incidents. Full complaint text and optional contact are restricted to staff by default. Public tracking shows a limited status/category/priority/area summary; anyone holding your tracking ID can see it. Aggregate counts are public. Optional contact does not trigger automatic SMS or email.', 'অনুমোদিত কর্মকর্তারা সমস্যা মূল্যায়ন, প্রস্তাবিত অগ্রাধিকার পর্যালোচনা ও সম্পর্কিত ঘটনা শনাক্ত করতে অভিযোগ ব্যবহার করেন। অভিযোগের সম্পূর্ণ লেখা ও ঐচ্ছিক যোগাযোগের তথ্য সাধারণভাবে শুধু কর্মকর্তারা দেখতে পান। ট্র্যাকিংয়ে সংক্ষিপ্ত অবস্থা, বিভাগ, অগ্রাধিকার ও এলাকা দেখা যায়; আইডি যার কাছে আছে তিনি তা দেখতে পারেন। মোট গণনার তথ্য প্রকাশ্য। যোগাযোগের তথ্য দিলে স্বয়ংক্রিয় এসএমএস বা ইমেইল পাঠানো হয় না।'),
    },
    {
      title: tr('Analysis and human review', 'বিশ্লেষণ ও মানব পর্যালোচনা'),
      body: tr('The backend performs rule/model analysis; this application does not send complaints to an external translation tool or generative AI service. Hosting, database and rate-limit providers process the necessary infrastructure data. Model suggestions can be wrong and staff can correct them. Priority is decision support, not a promise of response time or an emergency service.', 'ব্যাকএন্ডে নিয়ম ও মডেল দিয়ে বিশ্লেষণ হয়; এই অ্যাপ অভিযোগ বাইরের অনুবাদ বা জেনারেটিভ এআই সেবায় পাঠায় না। হোস্টিং, ডেটাবেস ও অনুরোধ সীমা নিয়ন্ত্রণকারী সেবা প্রয়োজনীয় অবকাঠামোগত তথ্য প্রক্রিয়া করে। মডেলের প্রস্তাব ভুল হতে পারে; কর্মকর্তারা তা সংশোধন করতে পারেন। অগ্রাধিকার সিদ্ধান্তে সহায়তা করে, নির্দিষ্ট সাড়া দেওয়ার সময় বা জরুরি সেবার প্রতিশ্রুতি নয়।'),
    },
    {
      title: tr('Browser storage and network records', 'ব্রাউজার ও নেটওয়ার্কের তথ্য'),
      body: tr('Only your interface-language preference is saved in local storage. Complaint drafts, contact details, passwords and administrator tokens are not saved there. Administrator tokens stay in memory until expiry, reload or sign-out. Rate limits use your network address; infrastructure providers may keep access logs. Keep your tracking ID private and close shared-device sessions.', 'লোকাল স্টোরেজে শুধু ইন্টারফেসের ভাষা সংরক্ষিত হয়। অভিযোগের খসড়া, যোগাযোগ, পাসওয়ার্ড বা প্রশাসকের টোকেন সেখানে রাখা হয় না। টোকেন মেয়াদ শেষ, রিলোড বা প্রস্থান পর্যন্ত মেমোরিতে থাকে। অনুরোধ সীমা নিয়ন্ত্রণে নেটওয়ার্ক ঠিকানা ব্যবহৃত হয়; অবকাঠামো সেবাদাতা প্রবেশের লগ রাখতে পারেন। ট্র্যাকিং আইডি গোপন রাখুন এবং যৌথ ডিভাইসে সেশন বন্ধ করুন।'),
    },
    {
      title: tr('Retention policy', 'কতদিন তথ্য রাখা হয়'),
      body: tr('Resolved or rejected complaints become eligible for reviewed deletion 90 days after their latest update. Authorised staff must preview and apply the deletion tool; there is no automatic schedule. This removes the selected complaint content, location, contact and stored analysis and redacts related review/audit evidence. Open cases are reviewed every 180 days. Other review/audit narratives have a reference limit of 180 days after closure; diagnostic logs, temporary feedback exports and rolling backups have a 30-day limit. A documented dispute or legal hold can extend retention. These are operator-managed targets, not an automatic deletion feature.', 'সমাধান করা বা প্রত্যাখ্যাত অভিযোগ সর্বশেষ হালনাগাদের ৯০ দিন পরে পর্যালোচনা করে মোছার উপযোগী হয়। অনুমোদিত কর্মকর্তাকে প্রথমে মোছার পরিধি দেখে তারপর প্রয়োগ করতে হয়; স্বয়ংক্রিয় সময়সূচি নেই। এতে নির্বাচিত অভিযোগের লেখা, এলাকা, যোগাযোগ ও সংরক্ষিত বিশ্লেষণ মুছে যায় এবং সম্পর্কিত পর্যালোচনা ও নিরীক্ষার ব্যক্তিগত তথ্য বাদ দেওয়া হয়। খোলা অভিযোগ ১৮০ দিন পরপর পর্যালোচনা করা হয়। অন্যান্য পর্যালোচনা ও নিরীক্ষার বিস্তারিত বন্ধ হওয়ার পর সর্বোচ্চ ১৮০ দিন, আর কারিগরি লগ, অস্থায়ী ফিডব্যাক রপ্তানি ও ঘূর্ণায়মান ব্যাকআপ সর্বোচ্চ ৩০ দিন রাখার নীতি রয়েছে। নথিভুক্ত বিরোধ বা আইনি সংরক্ষণে সময় বাড়তে পারে। এগুলো পরিচালকের দায়িত্বে থাকা লক্ষ্য; স্বয়ংক্রিয় তথ্য মোছার সুবিধা নয়।'),
    },
    {
      title: tr('Corrections, deletion and research use', 'সংশোধন, তথ্য মোছা ও গবেষণায় ব্যবহার'),
      body: tr('Contact the operator with your tracking ID to ask about access, correction or deletion. Do not send additional identity documents unless a proportionate verification method is agreed. The reference target is acknowledgement within 7 business days and a decision within 30 days. Related review text, exports, caches and backup expiry must also be considered; removing a database row alone is not complete deletion. Live complaints are not automatically approved research data. Research reuse needs separate permission, minimisation and review.', 'তথ্য দেখা, সংশোধন বা মোছার অনুরোধে ট্র্যাকিং আইডিসহ পরিচালকের সাথে যোগাযোগ করুন। উপযুক্ত যাচাই-পদ্ধতিতে সম্মতি ছাড়া বাড়তি পরিচয়পত্র পাঠাবেন না। নির্ধারিত লক্ষ্য ৭ কর্মদিবসে অনুরোধের প্রাপ্তি জানানো এবং ৩০ দিনের মধ্যে সিদ্ধান্ত দেওয়া। সম্পর্কিত পর্যালোচনা, রপ্তানি, ক্যাশ ও ব্যাকআপের মেয়াদও বিবেচনা করতে হবে; শুধু ডেটাবেসের একটি সারি মোছা সম্পূর্ণ অপসারণ নয়। সরাসরি জমা অভিযোগ স্বয়ংক্রিয়ভাবে গবেষণার তথ্য হয়ে যায় না। গবেষণায় ব্যবহারে আলাদা অনুমতি, তথ্য সীমিতকরণ ও পর্যালোচনা দরকার।'),
    },
  ]
  return <article className="tracking-layout">
    <div className="eyebrow"><Icon name="shield"/>{tr('Privacy & responsible use', 'গোপনীয়তা ও দায়িত্বশীল ব্যবহার')}</div>
    <h1>{tr('Your information, handled with care.', 'আপনার তথ্য, যত্নের সাথে।')}</h1>
    <p className="mt-4 text-sm leading-7">{tr('Notice version', 'নীতির সংস্করণ')}: {PRIVACY_VERSION}</p>
    {!operatorPublished && <div role="note" className="privacy-note mt-6"><p>{tr('Demo notice: this deployment has not published complete operator contact details. Use synthetic examples and do not submit personal or sensitive information until the operator has adopted this policy and supplied a privacy contact.', 'ডেমোর বিজ্ঞপ্তি: এই স্থাপনার পরিচালকের পূর্ণ যোগাযোগের তথ্য প্রকাশ করা হয়নি। পরিচালক এই নীতি গ্রহণ করে গোপনীয়তার যোগাযোগ প্রকাশ না করা পর্যন্ত কাল্পনিক উদাহরণ ব্যবহার করুন; ব্যক্তিগত বা সংবেদনশীল তথ্য দেবেন না।')}</p></div>}
    {operatorPublished && <div className="privacy-note mt-6"><p>{tr('Operated by', 'পরিচালনায়')}: <strong>{privacyOperator}</strong><br/><a className="text-button" href={privacyContact!} rel="noreferrer">{tr('Contact the privacy operator', 'গোপনীয়তার বিষয়ে যোগাযোগ করুন')}</a></p></div>}
    <div className="mt-8 space-y-7">{sections.map(section => <section key={section.title} className="rounded-2xl border border-teal-200 bg-white p-6"><h2 className="text-lg font-semibold">{section.title}</h2><p className="mt-3 text-sm leading-8">{section.body}</p></section>)}</div>
    <Link className="button-secondary mt-8" to="/">{tr('Back to the public portal', 'জনসেবা পোর্টালে ফিরুন')}</Link>
  </article>
}
