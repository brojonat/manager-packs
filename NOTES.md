## Original idea (pre agent-skills era):

ManagerPack: a collection of blog posts you (as a manager) can provide to junior
developers that will level them up. I write them with ChatGPT and resell them
under this product. Managers buy them as white labeled slide decks and present
them to their juniors. Kind of like what previous coworker Aaron did with Redis,
Futures, networking, ect. Similarly it could be a collection of projects that
demonstrate idiomatic language techniques. Managers would buy this like buying a
text book for their mentees. Alternatively, students buy them to jumpstart their
professional career. This is kinda just like Udemy with a twist? You could model
this with Temporal to demonstrate how to implement a subscription based
business. Actually that's idea: implement Udemy in Temporal with me being the
single provider of content. Document the process.

## Tweaked Idea (still pre agent-skills era but closer):

Circling back to ManagerPack, this is a good idea that I can absolute see being
valuable having worked at Hyundai. Think of this as a concentrated dose of
mentorship that managers give to the mentees as steriods. A collection of slide
decks and/or markdown documents and/or Looms. Let's also build the platform that
optimizes in finding what that is. It's basically Udemy implemented with
Temporal with an experimentation layer build in to it. The novel part is what is
the content? Traditionally, you'd have to have people generate content for you.
We can actually generate that content more or less upfront with an LLM and have
specialized knowledge dumps for various domains. We design this to conform
perfectly to our experimentation schema so that you can see exactly which
content users deem worth the price tag. This is relatively easy because we can
effectively just instrument links to Gumroad pages and surface the analytics on
top of that. We can also go so far as to make the content free and surface
content users gave money to nevertheless. Example content is slide decks,
markdown notes, curated lists, graphical views of data. I generate all the
content, all the payments go to me. Now the task is just to be creative!

## Final polish:

This is basically [skill.sh](https://skills.sh), but the payoff is not money for
me, it's Data Science group productivity. We have highly specific skills for the
following traditional data science tasks in our group. Typically each will
include a Marimo notebook demonstrating concepts in detail. We'll start with
just the institutional knowledge of our group. This will be like "playbooks" but
for Agent use, not only human use. We can also convert existing human only
materials into skills etc for agents. I will personally develop a skill for
this: "How to Unfuck Your Playbooks".

### Tabular Data Problems

- EDA on highly dimensional tabular data, PCA, etc
- EDA on image dataset with neural network
- Fine tuning local LLM
- Binary classification problem
- Multi class classification problem
- Multi label classification problem
- Regression problem
- Unsupervised learning

### Tech Stack

We'll put all of these into skills in this repo, but here's the tech stack from
an informal high level:

- Typr CLI
- MLFlow for experimentation tracking
- Scikit Learn for ML pipelines
- Ibis for data frames
- Marimo for notebooks w/ interactive Matplotlib widgets using
  [Anywidget](https://github.com/manzt/anywidget) and
  [Wigglystuff](https://github.com/koaning/wigglystuff)
